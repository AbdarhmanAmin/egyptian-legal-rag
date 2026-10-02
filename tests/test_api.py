from fastapi.testclient import TestClient

from src.api import main


def test_empty_question_returns_422():
    response = TestClient(main.app).post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_returns_article_citations(monkeypatch):
    monkeypatch.setattr(
        main,
        "run_rag_pipeline",
        lambda question: {
            "answer": "Answer",
            "sources": ["Egyptian Civil Code, Article 147"],
        },
    )
    response = TestClient(main.app).post(
        "/ask", json={"question": "What is the effect of a contract?"}
    )
    assert response.status_code == 200
    assert response.json()["sources"] == ["Egyptian Civil Code, Article 147"]
