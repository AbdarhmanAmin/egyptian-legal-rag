"""Prepare article-aware chunks while retaining article citations."""

import json
import re
import uuid
from pathlib import Path

import mlflow
import yaml


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[إأآ]", "ا", text).replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def split_article_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "chunk_size must be positive and overlap smaller than chunk_size"
        )
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        boundary = text.find(" ", next_start, end)
        if boundary >= 0:
            next_start = boundary + 1
        start = next_start

    return chunks


def build_article_chunks(
    articles: list[dict], chunk_size: int, overlap: int
) -> list[dict]:
    chunks = []
    for article in articles:
        number = int(article["article_number"])
        text_ar = normalize_arabic(article.get("text_ar", ""))
        if not text_ar:
            raise ValueError(f"Article {number} has no Arabic text")

        prefix = f"المادة {number}: "
        body_size = chunk_size - len(prefix)
        text_parts = split_article_text(
            text_ar, body_size, min(overlap, body_size - 1)
        )
        for index, text in enumerate(text_parts):
            chunk_text = f"{prefix}{text}"
            chunk_id = (
                number
                if len(text_parts) == 1
                else str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"article:{number}:chunk:{index}:size:{chunk_size}:"
                        f"overlap:{overlap}",
                    )
                )
            )
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        **article,
                        "text_ar": chunk_text,
                        "citation": f"Egyptian Civil Code, Article {number}",
                        "chunk_index": index,
                        "chunk_count": len(text_parts),
                        "is_repealed": bool(
                            article.get("is_repealed", False) or 54 <= number <= 80
                        ),
                    },
                }
            )
    return chunks


def run_chunking() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    source = Path(params["raw_data_path"])
    output = Path(params["processed_data_path"])
    articles = json.loads(source.read_text(encoding="utf-8"))
    chunking = params.get("chunking", {})
    chunk_size = int(chunking.get("chunk_size", 1536))
    overlap = int(chunking.get("overlap", 0))
    chunks = build_article_chunks(articles, chunk_size, overlap)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    repealed_article_count = sum(
        bool(
            article.get("is_repealed", False)
            or 54 <= int(article["article_number"]) <= 80
        )
        for article in articles
    )
    chunk_metrics = {
        "article_count": len(articles),
        "chunk_count": len(chunks),
        "repealed_article_count": repealed_article_count,
        "active_article_count": len(articles) - repealed_article_count,
        "chunk_size_characters": chunk_size,
        "overlap_characters": overlap,
        "average_chunks_per_article": round(len(chunks) / len(articles), 3)
        if articles
        else 0.0,
    }
    metrics_path = Path("reports/chunking_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(chunk_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="article_chunking"):
        mlflow.log_params(
            {
                "chunking_strategy": "article_aware",
                "chunk_size": chunk_size,
                "chunk_size_unit": "characters",
                "overlap": overlap,
                "embedding_model": params["embedding_model"],
            }
        )
        mlflow.log_metric("article_count", len(articles))
        mlflow.log_metric("chunk_count", len(chunks))
        mlflow.log_metric("repealed_article_count", repealed_article_count)

    print(f"Saved {len(chunks)} article chunks to {output}")


if __name__ == "__main__":
    run_chunking()
