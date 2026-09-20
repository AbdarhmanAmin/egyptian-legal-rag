import json
from pathlib import Path
import mlflow
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import yaml


def load_params():
  with open("params.yaml", "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def run_indexing():
  params = load_params()

  # 1. Load Data
  processed_data_path = Path(params["processed_data_path"])
  if not processed_data_path.exists():
    raise FileNotFoundError(
        f"Processed data file not found: {processed_data_path}"
    )

  with open(processed_data_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

  # 2. MLflow Setup
  mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
  mlflow.set_experiment(params["mlflow"]["experiment_name"])

  with mlflow.start_run(run_name="qdrant_indexing_stage"):
    # 3. Load Embedding Model
    model_name = params.get("embedding_model", "BAAI/bge-m3")
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Updated dimension check
    if hasattr(model, "get_embedding_dimension"):
      vector_size = model.get_embedding_dimension()
    else:
      vector_size = model.get_sentence_embedding_dimension()

    print(f"Model embedding dimension: {vector_size}")

    # 4. Initialize Qdrant Client
    qdrant_path = params.get("qdrant", {}).get("path", "./qdrant_storage")
    collection_name = params.get("qdrant", {}).get(
        "collection_name", "egyptian_civil_code"
    )

    client = QdrantClient(path=qdrant_path)

    # Safe Collection Re-creation using modern Qdrant methods
    if client.collection_exists(collection_name):
      print(f"Deleting existing collection '{collection_name}'...")
      client.delete_collection(collection_name)

    print(f"Creating Qdrant collection: {collection_name} (Dim: {vector_size})...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size, distance=Distance.COSINE
        ),
    )

    # 5. Generate Embeddings & Upsert to Qdrant
    points = []
    batch_size = 32

    print(f"Indexing {len(chunks)} chunks into Qdrant...")
    for i in range(0, len(chunks), batch_size):
      batch = chunks[i : i + batch_size]
      texts = [chunk["text"] for chunk in batch]

      # Generate embeddings for the batch
      embeddings = model.encode(texts, show_progress_bar=False)

      for chunk, embedding in zip(batch, embeddings):
        point = PointStruct(
            id=chunk["id"],
            vector=embedding.tolist(),
            payload=chunk["metadata"],
        )
        points.append(point)

      # Upsert batch into Qdrant
      client.upsert(
          collection_name=collection_name, points=points[-len(batch) :]
      )

    # 6. Log metrics to MLflow
    mlflow.log_params({
        "embedding_model": model_name,
        "vector_dimension": vector_size,
        "collection_name": collection_name,
        "qdrant_storage_path": qdrant_path,
    })
    mlflow.log_metrics({"total_indexed_points": len(points)})

    print(
        f"Indexing completed successfully! {len(points)} points indexed into"
        f" '{collection_name}'."
    )


if __name__ == "__main__":
  run_indexing()