"""Generate cited answers from retrieved Civil Code articles."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from groq import Groq

from src.rag.retriever import retrieve_relevant_docs

load_dotenv()

PROMPT = """أجب عن السؤال اعتماداً على النصوص القانونية أدناه فقط.
إذا لم تتضمن النصوص إجابة واضحة، فقل إن المستندات المتاحة لا تكفي للإجابة.
إذا كانت المادة ملغاة، فاذكر ذلك صراحة ولا تعرضها كحكم نافذ.
اكتب إجابة موجزة بنقاط، واذكر رقم كل مادة تستند إليها.

النصوص القانونية:
{context}

السؤال: {question}

الإجابة:"""


def format_context(points: list) -> tuple[str, list[str]]:
    blocks, sources = [], []
    for point in points:
        payload = point.payload or {}
        citation = (
            payload.get("citation")
            or f"Egyptian Civil Code, Article {payload.get('article_number')}"
        )
        status = "ملغاة" if payload.get("is_repealed") else "نافذة"
        blocks.append(f"{citation} ({status})\n{payload.get('text_ar', '')}")
        if citation not in sources:
            sources.append(citation)
    return "\n\n".join(blocks), sources


def run_rag_pipeline(question: str, top_k: int | None = None) -> dict:
    points = retrieve_relevant_docs(question, top_k=top_k)
    if not points:
        return {
            "answer": "لا توجد مواد مطابقة في المستندات المتاحة.",
            "sources": [],
            "contexts": [],
        }

    context, sources = format_context(points)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("Set GROQ_API_KEY in .env to enable answer generation.")

    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    config = params.get("generator", {})
    response = Groq(api_key=api_key).chat.completions.create(
        model=config.get("model_name", "openai/gpt-oss-120b"),
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(context=context, question=question),
            }
        ],
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 1024),
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "contexts": [
            point.payload.get("text_ar", "") for point in points if point.payload
        ],
    }
