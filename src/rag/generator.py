import os
from dotenv import load_dotenv
from groq import Groq
from src.rag.retriever import retrieve_relevant_docs
import yaml

load_dotenv()

PROMPT_TEMPLATE = """أنت مستشار قانوني خبير ومتخصص في القانون المدني المصري.
أجب على سؤال المستخدم بناءً فقط وحصرياً على المواد القانونية المتاحة أدناه.
إذا لم تجد الإجابة صريحة في النصوص المرفقة، أجب بوضوح: "لا توجد مادة قانونية تغطي هذا السؤال في المستندات المتاحة."

قواعد التنسيق الإلزامية للإجابة:
1. صغ الإجابة في نقاط واضحة ومنظمة (Bullet Points) مع استخدام العناوين البارزة.
2. يمنع منعاً باتاً استخدام الجداول (Tables) أو وسوم HTML مثل <br>.
3. اذكر رقم المادة القانونية بوضوح في بداية كل نقطة معتمدة.

المواد القانونية المتاحة:
{context}

السؤال:
{question}

الإجابة القانونية الموثقة:"""


def load_params():
  """قراءة الإعدادات من params.yaml"""
  with open("params.yaml", "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def format_context(retrieved_points: list) -> tuple[str, list[str]]:
  """استخراج وتنسيق الحقول المطلوبة فقط من الـ Payload لكل مادة مسترجعة."""
  context_blocks = []
  sources = []

  for point in retrieved_points:
    payload = point.payload or {}

    article_num = payload.get("article_number", "غير محدد")
    citation = (
        payload.get("citation")
        or f"المادة ({article_num}) من القانون المدني المصري"
    )
    topic = payload.get("topic", "")
    text_ar = (
        payload.get("text_ar")
        or payload.get("text")
        or payload.get("text_en")
        or ""
    )

    block = f"--- {citation} ---\nالموضوع: {topic}\nالنص: {text_ar}"
    context_blocks.append(block)

    if citation not in sources:
      sources.append(citation)

  return "\n\n".join(context_blocks), sources


def run_rag_pipeline(question: str) -> dict:
  """إدارة الـ Pipeline بالكامل: Hybrid Retrieval + LLM Generation عبر Groq"""
  # إعادة تحميل البيئة لضمان التقاط أي تغييرات جديدة في ملف .env
  load_dotenv(override=True)

  params = load_params()
  gen_config = params.get("generator", {})

  # 1. الاسترجاع الهجين من Qdrant عبر Retriever
  retrieved_points = retrieve_relevant_docs(question)

  if not retrieved_points:
    return {
        "answer": (
            "لا توجد مواد قانونية مطابقة في قاعدة البيانات للإجابة على هذا"
            " السؤال."
        ),
        "sources": [],
    }

  # 2. تجهيز الـ Context والـ Prompt
  context_str, sources = format_context(retrieved_points)
  prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)

  # 3. إعداد عميل Groq
  api_key = os.getenv("GROQ_API_KEY")
  if not api_key:
    raise ValueError(
        "لم يتم العثور على GROQ_API_KEY في ملف .env! يرجى التأكد من وجوده."
    )

  client = Groq(api_key=api_key)

  # 4. التوليد باستخدام الموديل المحدد في params.yaml
  model_name = gen_config.get("model_name", "openai/gpt-oss-120b")

  response = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt}],
      temperature=gen_config.get("temperature", 0.0),
      max_tokens=gen_config.get("max_tokens", 1024),
  )

  return {
      "answer": response.choices[0].message.content,
      "sources": sources,
  }


if __name__ == "__main__":
  test_question = "ما هي التزامات البائع في عقد البيع؟"
  print(f"السؤال: {test_question}\n")
  print("⚡ جاري الاسترجاع والتوليد عبر Groq...\n")

  result = run_rag_pipeline(test_question)

  print("=== الإجابة القانونية ===")
  print(result["answer"])
  print("\n=== المصادر المعتمدة ===")
  for src in result["sources"]:
    print(f"- {src}")