import json
from pathlib import Path

import pytest

CORPUS_PATH = Path("data/processed/egyptian_civil_code.json")
SOURCE_GAPS = {
    83,
    120,
    187,
    195,
    203,
    286,
    387,
    *range(389, 418),
    428,
    450,
    473,
    584,
    658,
    676,
    690,
    714,
    756,
    760,
    765,
    811,
    888,
    901,
    1022,
    1110,
    1115,
}


@pytest.fixture(scope="module")
def corpus():
    assert CORPUS_PATH.exists(), f"Missing corpus: {CORPUS_PATH}"
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def test_corpus_has_article_records(corpus):
    assert corpus
    assert all(isinstance(item["article_number"], int) for item in corpus)


def test_article_numbers_are_unique_and_sorted(corpus):
    numbers = [item["article_number"] for item in corpus]
    assert numbers == sorted(set(numbers))
    assert set(range(1, 1150)) - set(numbers) == SOURCE_GAPS


def test_required_fields_and_arabic_text(corpus):
    required = {
        "article_number",
        "book",
        "chapter",
        "section",
        "topic",
        "text_ar",
        "text_en",
        "is_repealed",
        "source_page",
        "citation",
    }
    for item in corpus:
        assert required <= item.keys()
        assert item["text_ar"].strip()
        assert any("\u0600" <= char <= "\u06ff" for char in item["text_ar"])
        assert len(item["text_ar"]) < 5000
        assert (
            item["citation"] == f"Egyptian Civil Code, Article {item['article_number']}"
        )


def test_repealed_articles_are_preserved_and_flagged(corpus):
    records = {item["article_number"]: item for item in corpus}
    for number in range(54, 81):
        assert records[number]["is_repealed"] is True
