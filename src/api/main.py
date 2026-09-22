from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

load_dotenv(override=True)

from src.api.schemas import AskRequest, AskResponse, HealthResponse
from src.rag.generator import run_rag_pipeline
from src.rag.retriever import (
    get_retriever_resources,  # <-- السطر المفقود الذي تسبب في الخطأ
)

# متغيّر لتخزين حالة التطبيق (مثل عدد المستندات المفهرسة)
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
  """إدارة دورة حياة التطبيق (Lifespan Context Manager):

  1. أثناء الـ Startup: نقوم بتحميل الموديلات والاتصال بـ Qdrant لمرة واحدة.
  2. نحسب عدد الـ Documents المفهرسة لخدمة الـ /health.
  3. أثناء الـ Shutdown: نغلق الموارد وننظف الذاكرة بنظافة.
  """
  print("🚀 جاري تشغيل سيرفر FastAPI...")
  print("⚡ تحضير الموديلات والاتصال بـ Qdrant (Cold Start)...")

  try:
    # 1. تحميل الموديلات والاتصال عبر الـ Cached Retriever
    client, dense_model, sparse_model, params = get_retriever_resources()
    collection_name = params.get("qdrant", {}).get(
        "collection_name", "egyptian_civil_code"
    )

    # 2. جلب عدد الـ Points المباشر من Qdrant لمطابقة شرط الـ Handbook
    collection_info = client.get_collection(collection_name)
    app_state["documents_indexed"] = collection_info.points_count
    print(
        "✅ تم الاتصال بـ Qdrant بنجاح! عدد المستندات المفهرسة:"
        f" {app_state['documents_indexed']}"
    )

  except Exception as e:
    print(f"❌ خطأ أثناء تهيئة الموارد عند الـ Startup: {e}")
    app_state["documents_indexed"] = 0

  yield  # هنا يعمل التطبيق ويستقبل الـ Requests

  # مرحلة الـ Shutdown
  print("🛑 إيقاف سيرفر FastAPI وتنظيف الموارد...")
  app_state.clear()


# إنشاء تطبيق FastAPI
app = FastAPI(
    title="Egyptian Legal RAG Assistant API",
    description="REST API لخدمة المساعد القانوني للقانون المدني المصري (Hybrid RAG)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System Status"])
async def health_check():
  """Endpoint للتحقق من سلامة النظام وعدد المستندات المفهرسة."""
  return HealthResponse(
      status="healthy",
      documents_indexed=app_state.get("documents_indexed", 0),
  )


@app.post("/ask", response_model=AskResponse, tags=["Legal RAG"])
async def ask_question(payload: AskRequest):
  """Endpoint لاستقبال الأسئلة القانونية وتوليد الإجابة الموثقة."""
  try:
    result = run_rag_pipeline(payload.question)
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
    )
  except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"حدث خطأ داخلي أثناء معالجة السؤال: {str(e)}",
    )