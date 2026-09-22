import json
from pathlib import Path
from fastembed import SparseTextEmbedding
import mlflow
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
import yaml


def load_params():
  with open("params.yaml", "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def run_indexing():
  params = load_params()

  processed_data_path = Path(params["processed_data_path"])
  if not processed_data_path.exists():
    raise FileNotFoundError(
        f"Processed data file not found: {processed_data_path}"
    )

  with open(processed_data_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

  # 1. تحميل نماذج الـ Embeddings
  print("Loading Dense (SentenceTransformer) and Sparse (BM25) models...")
  model_name = params.get("embedding_model", "BAAI/bge-m3")
  dense_model = SentenceTransformer(model_name)

  if hasattr(dense_model, "get_embedding_dimension"):
    vector_size = dense_model.get_embedding_dimension()
  else:
    vector_size = dense_model.get_sentence_embedding_dimension()

  sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

  # 2. إعداد Qdrant Client
  qdrant_path = params.get("qdrant", {}).get("path", "./qdrant_storage")
  collection_name = params.get("qdrant", {}).get(
      "collection_name", "egyptian_civil_code"
  )
  client = QdrantClient(path=qdrant_path)

  # إعادة إنشاء الـ Collection بإعدادات الـ Hybrid Search
  if client.collection_exists(collection_name):
    print(f"Deleting existing collection '{collection_name}'...")
    client.delete_collection(collection_name)

  print(
      f"Creating Qdrant collection: {collection_name} (Dense Dim:"
      f" {vector_size})..."
  )
  client.create_collection(
      collection_name=collection_name,
      vectors_config={
          "text-dense": VectorParams(size=vector_size, distance=Distance.COSINE)
      },
      sparse_vectors_config={"text-sparse": SparseVectorParams()},
  )

  # 3. استخراج الـ Vectors وتخزينها
  print(f"Indexing {len(chunks)} chunks using Hybrid Vectors...")
  texts = [chunk["text"] for chunk in chunks]

  # توليد المتجهات
  dense_embeddings = dense_model.encode(texts, show_progress_bar=True)
  sparse_embeddings = list(sparse_model.embed(texts))

  points = []
  for idx, (chunk, dense_emb, sparse_emb) in enumerate(
      zip(chunks, dense_embeddings, sparse_embeddings)
  ):
    point = PointStruct(
        id=chunk["id"],
        vector={
            "text-dense": dense_emb.tolist(),
            "text-sparse": SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            ),
        },
        payload=chunk["metadata"],
    )
    points.append(point)

  # Upsert إلى Qdrant
  client.upsert(collection_name=collection_name, points=points)
  print(
      f"Hybrid Indexing completed successfully! {len(points)} points indexed."
  )


if __name__ == "__main__":
  run_indexing()