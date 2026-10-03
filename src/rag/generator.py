"""Generate cited answers from retrieved Civil Code articles."""

import os
import re
from pathlib import Path
from typing import Callable

import yaml
from dotenv import load_dotenv
from groq import Groq

from src.rag.retriever import retrieve_relevant_docs

load_dotenv()

PROMPTS = {
    "ar": """أجب باللغة العربية عن السؤال اعتماداً على النصوص القانونية أدناه فقط.
إذا لم تتضمن النصوص إجابة واضحة، فقل إن المستندات المتاحة لا تكفي للإجابة.
إذا كانت المادة ملغاة، فاذكر ذلك صراحة ولا تعرضها كحكم نافذ.
اكتب إجابة موجزة بنقاط، واذكر رقم كل مادة تستند إليها. لا تستخدم Markdown bold.
أجب عن السؤال مباشرة، ولا تضف مواد أو موضوعات أخرى من السياق إلا إذا كانت لازمة للإجابة.

النصوص القانونية:
{context}

السؤال: {question}

الإجابة:""",
    "en": """Answer in English, matching the language of the question. Use only the legal text below.
If the text does not clearly answer the question, say that the available documents are insufficient.
Explicitly identify repealed articles and never present them as current law.
Give a concise answer in bullet points and cite every article number used. Translate or explain the Arabic legal text in clear English. Do not use Markdown bold.
Answer the question directly. Do not add unrelated articles or topics from the context.

Legal text:
{context}

Question: {question}

Answer:""",
}


def question_language(question: str) -> str:
    """Return Arabic when the question contains Arabic script, otherwise English."""
    return "ar" if re.search(r"[\u0600-\u06ff]", question) else "en"


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


def run_rag_pipeline(
    question: str,
    top_k: int | None = None,
    retrieve: Callable[[str, int | None], list] | None = None,
) -> dict:
    points = (
        retrieve(question, top_k)
        if retrieve is not None
        else retrieve_relevant_docs(question, top_k=top_k)
    )
    if not points:
        language = question_language(question)
        return {
            "answer": (
                "لا توجد مواد مطابقة في المستندات المتاحة."
                if language == "ar"
                else "No matching articles were found in the available documents."
            ),
            "sources": [],
            "contexts": [],
        }

    context, sources = format_context(points)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("Set GROQ_API_KEY in .env to enable answer generation.")

    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    config = params.get("generator", {})
    groq_client = Groq(api_key=api_key, max_retries=5, timeout=120)
    response = groq_client.chat.completions.create(
        model=config.get("model_name", "qwen/qwen3.8-27b"),
        messages=[
            {
                "role": "user",
                "content": PROMPTS[question_language(question)].format(
                    context=context, question=question
                ),
            }
        ],
        temperature=config.get("temperature", 0.0),
        reasoning_effort=config.get("reasoning_effort", "none"),
        max_tokens=config.get("max_tokens", 900),
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "contexts": [
            point.payload.get("text_ar", "") for point in points if point.payload
        ],
    }
