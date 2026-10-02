"""Load the local embedding model and retrieve matching Civil Code articles."""

from functools import lru_cache
from pathlib import Path

import torch
import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def load_params() -> dict:
    return yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_retriever_resources():
    params = load_params()
    client = QdrantClient(path=params["qdrant"]["path"])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = SentenceTransformer(params["embedding_model"], local_files_only=True)
    model.max_seq_length = 512
    return client, model, params


def retrieve_relevant_docs(query: str, top_k: int | None = None) -> list:
    client, model, params = get_retriever_resources()
    collection = params["qdrant"]["collection_name"]
    limit = top_k or params.get("retriever", {}).get("top_k", 5)
    result = client.query_points(
        collection_name=collection,
        query=model.encode(query).tolist(),
        limit=limit,
    )
    return result.points
