"""Compare article-aware chunking settings with RAGAS and track them in MLflow."""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from statistics import fmean

import mlflow
import yaml
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig
from sentence_transformers import SentenceTransformer

from src.rag.chunking import build_article_chunks, run_chunking
from src.rag.generator import run_rag_pipeline
from src.rag.indexing import run_indexing

CHUNKING_CONFIGS = [
    {"chunk_size": 256, "overlap": 0},
    {"chunk_size": 512, "overlap": 64},
    {"chunk_size": 768, "overlap": 128},
    {"chunk_size": 1024, "overlap": 128},
    {"chunk_size": 1536, "overlap": 0},
]
REGISTERED_MODEL_NAME = "EgyptianLegalRAG"
QUALITY_THRESHOLD = 0.75
RAGAS_METRIC_NAMES = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
)
ANSWER_RELEVANCY_STRICTNESS = 1
EVALUATION_PROMPT_VERSION = "arabic-grounded-verdict-v5-four-metrics"


class CompactFaithfulnessVerdict(BaseModel):
    verdict: int = Field(ge=0, le=1)


class CompactFaithfulnessOutput(BaseModel):
    statements: list[CompactFaithfulnessVerdict]


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Use the already-loaded local embedding model for RAGAS similarity."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


class LegalRAGModel(mlflow.pyfunc.PythonModel):
    """Model wrapper for the winning configuration in the production index."""

    def __init__(self, top_k: int, chunk_size: int, overlap: int):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap

    def predict(self, context, model_input, params=None):
        return [
            run_rag_pipeline(question, top_k=self.top_k)
            for question in model_input["question"].tolist()
        ]


def load_evaluation_data() -> tuple[list[dict], list[dict]]:
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    articles = json.loads(
        Path(params["raw_data_path"]).read_text(encoding="utf-8")
    )
    questions = json.loads(
        Path("data/evaluation/questions.json").read_text(encoding="utf-8")
    )
    by_number = {item["article_number"]: item for item in articles}
    for item in questions:
        article_number = item["article_number"]
        if article_number not in by_number:
            raise ValueError(
                f"Evaluation article {article_number} is not in the corpus"
            )
        item["ground_truth"] = by_number[article_number]["text_ar"]
    return articles, questions


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_completed_result(
    tracking_client: MlflowClient,
    experiment_id: str,
    run_name: str,
    report_path: Path,
    config: dict,
    questions: list[dict],
    model_name: str,
    embedding_model_name: str,
    top_k: int,
    data_version: str,
    max_tokens: int,
) -> dict | None:
    """Reuse a completed result when its saved inputs match this evaluation."""
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected_questions = [row["question"] for row in questions]
    if (
        report.get("chunk_size") != config["chunk_size"]
        or report.get("overlap") != config["overlap"]
        or report.get("embedding_model") != embedding_model_name
        or report.get("generation_model") != model_name
        or report.get("evaluation_model") != model_name
        or report.get("evaluation_prompt_version") != EVALUATION_PROMPT_VERSION
        or report.get("evaluation_max_tokens") != max_tokens
        or report.get("answer_relevancy_strictness")
        != ANSWER_RELEVANCY_STRICTNESS
        or report.get("top_k") != top_k
        or [row.get("question") for row in report.get("answers", [])]
        != expected_questions
    ):
        return None

    runs = tracking_client.search_runs(
        [experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        max_results=20,
    )
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        if (
            run.info.status == "FINISHED"
            and params.get("data_version") == data_version
            and params.get("generation_model") == model_name
            and params.get("evaluation_model") == model_name
            and params.get("evaluation_prompt_version")
            == EVALUATION_PROMPT_VERSION
            and params.get("evaluation_max_tokens") == str(max_tokens)
            and params.get("answer_relevancy_strictness")
            == str(ANSWER_RELEVANCY_STRICTNESS)
            and params.get("evaluation_questions") == str(len(questions))
            and all(
                metrics.get(metric_name) is not None
                and metrics[metric_name] == report.get("metrics", {}).get(metric_name)
                for metric_name in (*RAGAS_METRIC_NAMES, "retrieval_hit_at_k")
            )
        ):
            return {
                "run_id": run.info.run_id,
                "chunk_count": report["chunk_count"],
                **{
                    metric_name: metrics[metric_name]
                    for metric_name in (*RAGAS_METRIC_NAMES, "retrieval_hit_at_k")
                },
            }
    return None


def collect_rag_results(
    questions: list[dict],
    client: QdrantClient,
    collection_name: str,
    top_k: int,
    query_vectors: list,
) -> tuple[Dataset, list[dict]]:
    answers = []
    for question_index, row in enumerate(questions, start=1):
        query_vector = query_vectors[question_index - 1]

        def retrieve(_: str, limit: int | None) -> list:
            result = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit or top_k,
            )
            return result.points

        result = run_rag_pipeline(row["question"], top_k=top_k, retrieve=retrieve)
        answers.append(
            {
                "question": row["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": row["ground_truth"],
            }
        )
        print(
            f"Generated answer {question_index}/{len(questions)}",
            flush=True,
        )

    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in answers],
            "answer": [row["answer"] for row in answers],
            "contexts": [row["contexts"] for row in answers],
            "ground_truth": [row["ground_truth"] for row in answers],
        }
    )
    return dataset, answers


def load_answer_cache(
    cache_path: Path,
    config: dict,
    questions: list[dict],
    embedding_model_name: str,
    generation_model: str,
    top_k: int,
    data_version: str,
) -> list[dict] | None:
    """Reuse generated answers after an interrupted RAGAS scoring run."""
    if not cache_path.exists():
        return None
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_questions = [row["question"] for row in questions]
    if (
        cache.get("chunk_size") != config["chunk_size"]
        or cache.get("overlap") != config["overlap"]
        or cache.get("embedding_model") != embedding_model_name
        or cache.get("generation_model") != generation_model
        or cache.get("top_k") != top_k
        or cache.get("data_version") != data_version
        or [row.get("question") for row in cache.get("answers", [])]
        != expected_questions
    ):
        return None
    return cache["answers"]


def make_dataset(answer_rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [row["question"] for row in answer_rows],
            "answer": [row["answer"] for row in answer_rows],
            "contexts": [row["contexts"] for row in answer_rows],
            "ground_truth": [row["ground_truth"] for row in answer_rows],
        }
    )


def prepare_evaluation_rows(
    answer_rows: list[dict],
    questions: list[dict],
    articles: list[dict],
) -> list[dict]:
    """Give RAGAS the same cited, status-labeled context the generator received."""
    articles_by_number = {int(row["article_number"]): row for row in articles}
    prepared = []
    for answer_row, question_row in zip(answer_rows, questions):
        blocks = []
        retrieved_numbers = []
        for raw_context in answer_row.get("contexts", []):
            match = re.match(r"\s*المادة\s+(\d+)\s*:", raw_context)
            if match is None:
                blocks.append(raw_context)
                continue

            number = int(match.group(1))
            article = articles_by_number.get(number, {})
            is_repealed = bool(
                article.get("is_repealed", False) or 54 <= number <= 80
            )
            status = "ملغاة" if is_repealed else "نافذة"
            citation = article.get(
                "citation", f"Egyptian Civil Code, Article {number}"
            )
            blocks.append(f"{citation} ({status})\n{raw_context}")
            if number not in retrieved_numbers:
                retrieved_numbers.append(number)

        target_number = int(question_row["article_number"])
        prepared.append(
            {
                **answer_row,
                "contexts": blocks,
                "article_number": target_number,
                "retrieved_article_numbers": retrieved_numbers,
                "expected_article_retrieved": target_number in retrieved_numbers,
            }
        )
    return prepared


def hide_stale_evaluation_runs(
    tracking_client: MlflowClient,
    experiment_id: str,
    retained_run_ids: set[str],
) -> int:
    """Keep the newest completed five-run comparison visible in MLflow."""
    stale_runs = [
        run
        for run in tracking_client.search_runs([experiment_id], max_results=1000)
        if run.info.run_name.startswith("article_chunk_size_")
        and run.info.run_id not in retained_run_ids
    ]
    for run in stale_runs:
        tracking_client.delete_run(run.info.run_id)
    return len(stale_runs)


def run_experiments() -> None:
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "RAGAS evaluation requires Python 3.11 in this project. "
            "Run .venv311\\Scripts\\python.exe -m "
            "src.experiments.evaluate_rag instead."
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("Set GROQ_API_KEY before running RAGAS experiments")

    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")
    tracking_client = MlflowClient()

    articles, questions = load_evaluation_data()
    question_limit = int(params.get("evaluation", {}).get("question_limit", 5))
    if question_limit < 1:
        raise ValueError("evaluation.question_limit must be at least 1")
    if len(questions) > question_limit:
        selected = sorted(
            random.Random(42).sample(range(len(questions)), question_limit)
        )
        questions = [questions[index] for index in selected]

    model_name = params["embedding_model"]
    top_k = int(params.get("retriever", {}).get("top_k", 7))
    source_hash = hashlib.sha256(
        Path(params["raw_data_path"]).read_bytes()
    ).hexdigest()
    try:
        embedding_model = SentenceTransformer(model_name, local_files_only=True)
    except TypeError:
        embedding_model = SentenceTransformer(model_name)
    embedding_model.max_seq_length = 512

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            model=params["generator"]["model_name"],
            api_key=api_key,
            temperature=0,
            reasoning_effort=params["generator"].get("reasoning_effort", "none"),
            max_tokens=int(params.get("evaluation", {}).get("max_tokens", 500)),
            max_retries=2,
            timeout=60,
        )
    )
    evaluator_embeddings = LocalSentenceTransformerEmbeddings(embedding_model)
    evaluation_max_tokens = int(
        params.get("evaluation", {}).get("max_tokens", 500)
    )
    faithfulness.statement_generator_prompt.instruction = (
        "استخرج الادعاءات القانونية الموجودة في الإجابة فقط. "
        "قسّم الإجابة إلى عبارات واقعية مستقلة ومختصرة، دون إضافة معلومات. "
        "أعد JSON المطلوب فقط."
    )
    faithfulness.statement_generator_prompt.examples = []
    faithfulness.nli_statements_prompt.instruction = (
        "قيّم كل عبارة اعتمادًا على النصوص القانونية المرفقة فقط. "
        "اعتبر العبارة مدعومة إذا كان معناها القانوني مطابقًا أو مستفادًا بوضوح "
        "من النص، حتى لو اختلفت الصياغة أو احتوى النص على أخطاء OCR. "
        "اكتب verdict=1 للعبارة المدعومة وverdict=0 للعبارة التي تضيف حكمًا "
        "غير موجود في السياق. "
        "أعد حكمًا واحدًا لكل عبارة وبالترتيب نفسه، دون شرح أو استنتاج. "
        "أعد JSON المطلوب فقط."
    )
    faithfulness.nli_statements_prompt.examples = []
    faithfulness.nli_statements_prompt.output_model = CompactFaithfulnessOutput
    context_precision.context_precision_prompt.instruction = (
        "قيّم هذا السياق القانوني وحده بالنسبة إلى السؤال والمرجع القانوني. "
        "اكتب verdict=1 إذا كان السياق يساعد في دعم المرجع أو يتناول حكمًا "
        "مرتبطًا مباشرة بالسؤال، وverdict=0 إذا كان غير ذي صلة. "
        "اكتب سببًا قصيرًا وأعد JSON المطلوب فقط."
    )
    context_precision.context_precision_prompt.examples = []
    context_recall.context_recall_prompt.instruction = (
        "قسّم المرجع القانوني إلى أحكام أو ادعاءات قانونية مستقلة. "
        "لكل ادعاء، اكتب attributed=1 إذا كانت النصوص المسترجعة تسنده "
        "بالمعنى القانوني، حتى مع اختلاف الصياغة أو وجود أخطاء OCR؛ "
        "وإلا اكتب attributed=0. اكتب سببًا قصيرًا لكل حكم وأعد JSON فقط."
    )
    context_recall.context_recall_prompt.examples = []
    answer_relevancy.question_generation.instruction = (
        "اكتب سؤالًا عربيًا موجزًا واحدًا يمكن الإجابة عنه اعتمادًا على "
        "الإجابة المقدمة. حدّد noncommittal=1 فقط إذا كانت الإجابة مراوغة "
        "أو غير حاسمة، وإلا فاكتب 0. أعد JSON المطلوب فقط."
    )
    answer_relevancy.strictness = ANSWER_RELEVANCY_STRICTNESS

    mlflow_client = QdrantClient(":memory:")
    summary = []
    retained_run_ids = set()
    report_dir = Path("reports/ragas_runs")
    report_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    print(
        f"Evaluating {len(CHUNKING_CONFIGS)} chunking configurations with "
        f"{len(questions)} questions each.",
        flush=True,
    )
    query_vectors = None

    for config in CHUNKING_CONFIGS:
        chunk_size = config["chunk_size"]
        overlap = config["overlap"]
        run_name = f"article_chunk_size_{chunk_size}_overlap_{overlap}"
        report_path = report_dir / f"{run_name}.json"
        completed = load_completed_result(
            tracking_client,
            experiment.experiment_id,
            run_name,
            report_path,
            config,
            questions,
            params["generator"]["model_name"],
            model_name,
            top_k,
            source_hash,
            evaluation_max_tokens,
        )
        if completed is not None:
            retained_run_ids.add(completed.pop("run_id"))
            summary.append({**config, **completed})
            print(
                f"{run_name}: reusing completed MLflow result {completed}",
                flush=True,
            )
            continue

        print(f"Starting {run_name}", flush=True)
        answer_cache_path = report_dir / f"{run_name}.answers.json"
        answer_rows = load_answer_cache(
            answer_cache_path,
            config,
            questions,
            model_name,
            params["generator"]["model_name"],
            top_k,
            source_hash,
        )
        chunks = build_article_chunks(articles, chunk_size, overlap)

        if answer_rows is None:
            if query_vectors is None:
                query_vectors = embedding_model.encode(
                    [row["question"] for row in questions], batch_size=32
                ).tolist()
            collection_name = f"chunk_size_{chunk_size}_overlap_{overlap}"
            mlflow_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_model.get_embedding_dimension(),
                    distance=Distance.COSINE,
                ),
            )
            vectors = embedding_model.encode(
                [chunk["text"] for chunk in chunks], batch_size=32
            )
            mlflow_client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=chunk["id"],
                        vector=vector.tolist(),
                        payload=chunk["metadata"],
                    )
                    for chunk, vector in zip(chunks, vectors)
                ],
            )
            _, answer_rows = collect_rag_results(
                questions,
                mlflow_client,
                collection_name,
                top_k,
                query_vectors,
            )
            answer_cache_path.write_text(
                json.dumps(
                    {
                        **config,
                        "embedding_model": model_name,
                        "generation_model": params["generator"]["model_name"],
                        "top_k": top_k,
                        "data_version": source_hash,
                        "answers": answer_rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            print(f"Reusing saved answers for {run_name}", flush=True)
        answer_rows = prepare_evaluation_rows(answer_rows, questions, articles)
        dataset = make_dataset(answer_rows)
        with mlflow.start_run(run_name=run_name) as active_run:
            current_run_id = active_run.info.run_id
            mlflow.log_params(
                {
                    "chunking_strategy": "article_aware",
                    "chunk_size": chunk_size,
                    "chunk_size_unit": "characters",
                    "overlap": overlap,
                    "embedding_model": model_name,
                    "generation_model": params["generator"]["model_name"],
                    "evaluation_model": params["generator"]["model_name"],
                    "evaluation_prompt_version": EVALUATION_PROMPT_VERSION,
                    "evaluation_max_tokens": evaluation_max_tokens,
                    "answer_relevancy_strictness": answer_relevancy.strictness,
                    "top_k": top_k,
                    "evaluation_questions": len(questions),
                    "data_version": source_hash,
                }
            )
            mlflow.set_tags(
                {
                    "git_commit": commit,
                    "data_version": source_hash,
                    "framework": "ragas",
                }
            )
            scores = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    context_precision,
                    context_recall,
                    answer_relevancy,
                ],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=RunConfig(
                    timeout=120,
                    max_retries=int(
                        params.get("evaluation", {}).get("max_retries", 2)
                    ),
                    max_wait=8,
                    max_workers=1,
                ),
                raise_exceptions=True,
            )
            per_metric_scores = {
                metric_name: list(map(float, scores[metric_name]))
                for metric_name in RAGAS_METRIC_NAMES
            }
            metrics = {
                metric_name: fmean(question_scores)
                for metric_name, question_scores in per_metric_scores.items()
            }
            metrics["retrieval_hit_at_k"] = fmean(
                float(row["expected_article_retrieved"]) for row in answer_rows
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("data/evaluation/questions.json", artifact_path="data")
            mlflow.log_artifact("pyproject.toml", artifact_path="environment")

            run_report = {
                **config,
                "embedding_model": model_name,
                "generation_model": params["generator"]["model_name"],
                "evaluation_model": params["generator"]["model_name"],
                "evaluation_prompt_version": EVALUATION_PROMPT_VERSION,
                "evaluation_max_tokens": evaluation_max_tokens,
                "answer_relevancy_strictness": answer_relevancy.strictness,
                "top_k": top_k,
                "question_count": len(questions),
                "chunk_count": len(chunks),
                "metrics": metrics,
                "per_question_faithfulness": [
                    {
                        "question": row["question"],
                        "score": per_metric_scores["faithfulness"][index],
                    }
                    for index, row in enumerate(answer_rows)
                ],
                "per_question_metrics": [
                    {
                        "question": row["question"],
                        **{
                            metric_name: per_metric_scores[metric_name][index]
                            for metric_name in RAGAS_METRIC_NAMES
                        },
                        "expected_article_retrieved": row[
                            "expected_article_retrieved"
                        ],
                    }
                    for index, row in enumerate(answer_rows)
                ],
                "retrieval_hit_at_k": metrics["retrieval_hit_at_k"],
                "answers": answer_rows,
            }
            report_path.write_text(
                json.dumps(run_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            summary.append({**config, "chunk_count": len(chunks), **metrics})
            print(f"{run_name}: {metrics}")
        retained_run_ids.add(current_run_id)

    hidden_count = hide_stale_evaluation_runs(
        tracking_client, experiment.experiment_id, retained_run_ids
    )
    if hidden_count:
        print(f"Archived {hidden_count} older MLflow evaluation runs.", flush=True)
    mlflow_client.close()
    Path("reports").mkdir(exist_ok=True)
    summary_path = Path("reports/ragas_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if len(questions) < 20:
        print(
            f"Saved {len(summary)} MLflow evaluation runs using {len(questions)} "
            "questions each. Model promotion is skipped; use at least 20 questions "
            "for the quality gate."
        )
        return

    best = max(
        summary,
        key=lambda item: item["faithfulness"],
    )
    if best["faithfulness"] < QUALITY_THRESHOLD:
        raise ValueError(
            f"Faithfulness quality gate failed: {best['faithfulness']:.3f} "
            f"< {QUALITY_THRESHOLD:.2f}; no configuration was promoted"
        )

    # Rebuild the production corpus and vector index with the selected settings.
    params["chunking"] = {
        "chunk_size": best["chunk_size"],
        "overlap": best["overlap"],
    }
    Path("params.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    run_chunking()
    run_indexing()

    with mlflow.start_run(run_name="best_article_aware_chunking"):
        mlflow.log_params(
            {
                "chunking_strategy": "article_aware",
                "chunk_size": best["chunk_size"],
                "chunk_size_unit": "characters",
                "overlap": best["overlap"],
                "embedding_model": model_name,
                "generation_model": params["generator"]["model_name"],
                "evaluation_model": params["generator"]["model_name"],
                "top_k": top_k,
                "evaluation_questions": len(questions),
            }
        )
        mlflow.log_metrics(
            {"faithfulness": best["faithfulness"]}
        )
        mlflow.log_artifact(str(summary_path), artifact_path="evaluation")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LegalRAGModel(
                top_k,
                best["chunk_size"],
                best["overlap"],
            ),
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    registered_version = model_info.registered_model_version
    if registered_version is None:
        raise RuntimeError("MLflow did not return the registered model version")
    mlflow_client = MlflowClient()
    mlflow_client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_version,
    )
    mlflow_client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=registered_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(
        f"Promoted {REGISTERED_MODEL_NAME} version {registered_version} to "
        f"Production (faithfulness={best['faithfulness']:.3f}, "
        f"chunk_size={best['chunk_size']}, overlap={best['overlap']})."
    )


if __name__ == "__main__":
    run_experiments()
