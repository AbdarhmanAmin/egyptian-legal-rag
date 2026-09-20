from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. الاتصال بالفولدر المحلي
client = QdrantClient(path="./qdrant_storage")
model = SentenceTransformer("BAAI/bge-m3")

# 2. سؤال قانوني تجريبي
query = "ما هي التزامات البائع في عقد البيع؟"
query_vector = model.encode(query).tolist()

# 3. البحث باستخدام query_points (Modern Qdrant API)
response = client.query_points(
    collection_name="egyptian_civil_code", query=query_vector, limit=3
)

print("\n=== نتائج البحث من Qdrant المحلي ===")
for r in response.points:
  print(
      f"\n- المادة ({r.payload.get('article_number')}) | Score:"
      f" {r.score:.4f}"
  )
  print(f"  النص: {r.payload.get('text_ar')}")