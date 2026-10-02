"""Build the local Qdrant index from article chunks."""

import json
from pathlib import Path

import torch
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def run_indexing() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    chunks_path = Path(params["processed_data_path"])
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    model_path = params["embedding_model"]
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = SentenceTransformer(model_path, local_files_only=True)
    model.max_seq_length = 512
    vectors = model.encode(
        [chunk["text"] for chunk in chunks], batch_size=32, show_progress_bar=True
    )

    qdrant = params["qdrant"]
    client = QdrantClient(path=qdrant["path"])
    collection = qdrant["collection_name"]
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE
        ),
    )
    points = [
        PointStruct(id=chunk["id"], vector=vector.tolist(), payload=chunk["metadata"])
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} articles in '{collection}'")


if __name__ == "__main__":
    run_indexing()
