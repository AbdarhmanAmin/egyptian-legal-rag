from types import SimpleNamespace

import numpy as np

from src.rag import retriever


def test_retriever_embeds_query_and_returns_qdrant_points(monkeypatch):
    expected = [SimpleNamespace(payload={"article_number": 147})]

    class FakeModel:
        def encode(self, query):
            assert query == "ما أثر العقد؟"
            return np.array([0.1, 0.2])

    class FakeClient:
        def query_points(self, **kwargs):
            assert kwargs["collection_name"] == "civil-code"
            assert kwargs["query"] == [0.1, 0.2]
            assert kwargs["limit"] == 3
            return SimpleNamespace(points=expected)

    monkeypatch.setattr(
        retriever,
        "get_retriever_resources",
        lambda: (
            FakeClient(),
            FakeModel(),
            {
                "qdrant": {"collection_name": "civil-code"},
                "retriever": {"top_k": 3},
            },
        ),
    )

    assert retriever.retrieve_relevant_docs("ما أثر العقد؟") == expected
