from functools import lru_cache
import os
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector
from sentence_transformers import SentenceTransformer
import yaml

load_dotenv()


def load_params():
  with open("params.yaml", "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


# تحميل الموديلات والعميل مرة واحدة فقط في الذاكرة Caching
@lru_cache(maxsize=1)
def get_retriever_resources():
  """تحميل نماذج الـ Embeddings والاتصال بـ Qdrant مرة واحدة فقط عند بدء التشغيل."""
  params = load_params()
  qdrant_path = params.get("qdrant", {}).get("path", "./qdrant_storage")
  embedding_model_name = params.get("embedding_model", "BAAI/bge-m3")

  print("⚡ Loading models into memory (Cold Start)...")
  client = QdrantClient(path=qdrant_path)
  dense_model = SentenceTransformer(embedding_model_name)
  sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

  return client, dense_model, sparse_model, params


def retrieve_relevant_docs(query: str, top_k: int = None) -> list:
  """استرجاع المواد باستخدام الموديلات المحملة مسبقاً."""
  client, dense_model, sparse_model, params = get_retriever_resources()

  if top_k is None:
    top_k = params.get("retriever", {}).get("top_k", 3)

  collection_name = params.get("qdrant", {}).get(
      "collection_name", "egyptian_civil_code"
  )

  # تحويل الاستعلام بسرعة عالية لأن الموديل جاهز بالـ RAM
  dense_query = dense_model.encode(query).tolist()
  sparse_query = list(sparse_model.embed([query]))[0]

  response = client.query_points(
      collection_name=collection_name,
      prefetch=[
          Prefetch(using="text-dense", query=dense_query, limit=top_k),
          Prefetch(
              using="text-sparse",
              query=SparseVector(
                  indices=sparse_query.indices.tolist(),
                  values=sparse_query.values.tolist(),
              ),
              limit=top_k,
          ),
      ],
      query=FusionQuery(fusion=Fusion.RRF),
      limit=top_k,
  )

  return response.points