import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import mlflow

from src.rag.chunking import article_aware_chunking, load_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_chunks(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    """Save the output chunks to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def generate_distribution_plot(lengths: List[int], output_path: Path, title: str) -> None:
    """Generate and save chunk length distribution plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=30, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("Length in Characters")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_experiment(
    input_path: Path,
    output_path: Path,
    strategy_name: str = "article_aware",
    max_chars: int = 1500,
) -> None:
    """Run chunking strategy and log results, metrics, and artifacts to MLflow."""
    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Egyptian_Legal_RAG_Chunking")

    corpus = load_corpus(input_path)
    chunks = article_aware_chunking(corpus, max_chars_per_chunk=max_chars)

    save_chunks(chunks, output_path)

    lengths = [len(c["text"]) for c in chunks]
    total_chunks = len(chunks)
    avg_len = sum(lengths) / total_chunks if total_chunks > 0 else 0
    min_len = min(lengths) if total_chunks > 0 else 0
    max_len = max(lengths) if total_chunks > 0 else 0

    plot_path = Path("reports/chunk_distribution.png")
    generate_distribution_plot(
        lengths,
        plot_path,
        f"Chunk Length Distribution ({strategy_name})",
    )

    with mlflow.start_run(run_name=f"run_{strategy_name}"):
        mlflow.log_param("strategy", strategy_name)
        mlflow.log_param("max_chars_per_chunk", max_chars)

        mlflow.log_metric("total_chunks", total_chunks)
        mlflow.log_metric("avg_chunk_length_chars", avg_len)
        mlflow.log_metric("min_chunk_length_chars", min_len)
        mlflow.log_metric("max_chunk_length_chars", max_len)

        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(output_path))

    logger.info("Experiment successfully completed and logged to MLflow.")


if __name__ == "__main__":
    input_file = Path("data/processed/egyptian_civil_code.json")
    output_file = Path("data/processed/chunks_article_aware.json")

    run_experiment(
        input_path=input_file,
        output_path=output_file,
        strategy_name="article_aware",
        max_chars=1500,
    )