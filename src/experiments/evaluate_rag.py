"""Compare real retrieval settings with RAGAS and record results in MLflow."""

import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any

import mlflow
import yaml
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from mlflow.tracking import MlflowClient
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.rag.generator import run_rag_pipeline

TOP_K_VALUES = [3, 5, 7, 10, 12]
MODEL_NAME = "BAAI/bge-m3"
REGISTERED_MODEL_NAME = "EgyptianLegalRAG"


class LegalRAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self, top_k: int):
        self.top_k = top_k

    def predict(
        self, context: Any, model_input: Any, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return [
            run_rag_pipeline(question, top_k=self.top_k)
            for question in model_input["question"].tolist()
        ]


def load_evaluation_set() -> list[dict]:
    questions = json.loads(
        Path("data/evaluation/questions.json").read_text(encoding="utf-8")
    )
    articles = json.loads(
        Path("data/processed/egyptian_civil_code.json").read_text(encoding="utf-8")
    )
    by_number = {item["article_number"]: item for item in articles}
    for item in questions:
        if item["article_number"] not in by_number:
            raise ValueError(
                f"Evaluation article {item['article_number']} is not in the corpus"
            )
        item["ground_truth"] = by_number[item["article_number"]]["text_ar"]
    return questions


def run_experiments() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("Set GROQ_API_KEY before running RAGAS experiments")

    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    dataset_rows = load_evaluation_set()

    eval_llm = LangchainLLMWrapper(
        ChatGroq(
            model=params["generator"]["model_name"], api_key=api_key, temperature=0
        )
    )
    local_embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"local_files_only": True},
    )
    eval_embeddings = LangchainEmbeddingsWrapper(local_embeddings)
    summary = []

    for top_k in TOP_K_VALUES:
        questions, answers, contexts, references = [], [], [], []
        for row in dataset_rows:
            result = run_rag_pipeline(row["question"], top_k=top_k)
            questions.append(row["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            references.append(row["ground_truth"])

        evaluation = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": references,
            }
        )
        with mlflow.start_run(run_name=f"article_chunks_top_k_{top_k}"):
            mlflow.log_params(
                {
                    "chunk_size": "article",
                    "overlap": 0,
                    "embedding_model": MODEL_NAME,
                    "top_k": top_k,
                    "evaluation_questions": len(questions),
                }
            )
            scores = evaluate(
                dataset=evaluation,
                metrics=[
                    faithfulness,
                    context_precision,
                    context_recall,
                    answer_relevancy,
                ],
                llm=eval_llm,
                embeddings=eval_embeddings,
            )
            metrics = {
                name: fmean(map(float, scores[name]))
                for name in (
                    "faithfulness",
                    "context_precision",
                    "context_recall",
                    "answer_relevancy",
                )
            }
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("data/evaluation/questions.json")
            summary.append({"top_k": top_k, **metrics})
            print(f"top_k={top_k}: {metrics}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/ragas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = max(summary, key=lambda item: item["faithfulness"])
    if best["faithfulness"] < 0.75:
        raise ValueError(f"Faithfulness gate failed: {best['faithfulness']:.3f} < 0.75")

    with mlflow.start_run(run_name=f"best_config_top_k_{best['top_k']}"):
        mlflow.log_params(
            {
                "chunk_size": "article",
                "overlap": 0,
                "embedding_model": MODEL_NAME,
                "top_k": best["top_k"],
            }
        )
        mlflow.log_metrics(
            {key: value for key, value in best.items() if key != "top_k"}
        )
        mlflow.log_artifact("reports/ragas_summary.json")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LegalRAGModel(best["top_k"]),
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        MlflowClient().set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=model_info.registered_model_version,
        )

    params["retriever"]["top_k"] = best["top_k"]
    Path("params.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Best top_k={best['top_k']} with faithfulness={best['faithfulness']:.3f}")


if __name__ == "__main__":
    run_experiments()
