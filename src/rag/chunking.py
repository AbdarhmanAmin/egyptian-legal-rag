"""Validate and save one searchable chunk for each legal article."""

import json
import re
from pathlib import Path

import mlflow
import yaml


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[إأآ]", "ا", text).replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def run_chunking() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    source = Path(params["raw_data_path"])
    output = Path(params["processed_data_path"])
    articles = json.loads(source.read_text(encoding="utf-8"))
    chunks = []

    for article in articles:
        number = int(article["article_number"])
        text_ar = normalize_arabic(article.get("text_ar", ""))
        if not text_ar:
            raise ValueError(f"Article {number} has no Arabic text")
        chunks.append(
            {
                "id": number,
                "text": f"المادة {number}: {text_ar}",
                "metadata": {
                    **article,
                    "text_ar": text_ar,
                    "citation": f"Egyptian Civil Code, Article {number}",
                    "is_repealed": bool(
                        article.get("is_repealed", False) or 54 <= number <= 80
                    ),
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="article_chunking"):
        mlflow.log_params(
            {
                "chunking_strategy": "one_article_per_chunk",
                "chunk_size": "article",
                "overlap": 0,
                "embedding_model": params["embedding_model"],
            }
        )
        mlflow.log_metric("article_count", len(chunks))
        mlflow.log_metric(
            "repealed_article_count", sum(c["metadata"]["is_repealed"] for c in chunks)
        )

    print(f"Saved {len(chunks)} article chunks to {output}")


if __name__ == "__main__":
    run_chunking()
