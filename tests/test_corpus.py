import json
import pytest
from pathlib import Path

CORPUS_PATH = Path("data/processed/egyptian_civil_code.json")

@pytest.fixture(scope="module")
def corpus():
    """Load the generated JSON corpus file."""
    assert CORPUS_PATH.exists(), f"File not found: {CORPUS_PATH}"
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Corpus JSON root must be a list"
    assert len(data) > 0, "Corpus JSON cannot be empty"
    return data

def test_exact_schema_keys(corpus):
    """Verify each record contains the exact required keys."""
    expected_keys = {
        "article_number",
        "book",
        "chapter",
        "section",
        "topic",
        "text_ar",
        "text_en",
        "is_repealed",
        "source_page",
        "citation"
    }
    for item in corpus:
        assert set(item.keys()) == expected_keys, f"Key mismatch in article {item.get('article_number')}"

def test_data_types(corpus):
    """Verify data types for every record field."""
    for item in corpus:
        assert isinstance(item["article_number"], int)
        assert isinstance(item["book"], str)
        assert isinstance(item["chapter"], str)
        assert isinstance(item["section"], str)
        assert isinstance(item["topic"], str)
        assert isinstance(item["text_ar"], str)
        assert isinstance(item["text_en"], str)
        assert isinstance(item["is_repealed"], bool)
        assert isinstance(item["source_page"], int)
        assert isinstance(item["citation"], str)

def test_non_empty_active_articles(corpus):
    """Ensure active articles have non-empty Arabic text."""
    for item in corpus:
        if not item["is_repealed"]:
            assert len(item["text_ar"].strip()) > 0, f"Article {item['article_number']} has empty Arabic text"

def test_repealed_articles_flagged(corpus):
    """Ensure repealed range (Articles 54 to 80) is flagged correctly."""
    repealed = [item for item in corpus if 54 <= item["article_number"] <= 80]
    assert len(repealed) > 0, "Repealed range (Articles 54-80) not found"
    for item in repealed:
        assert item["is_repealed"] is True, f"Article {item['article_number']} should have is_repealed=True"

def test_citation_formatting(corpus):
    """Check citation string pattern."""
    for item in corpus:
        expected = f"Egyptian Civil Code, Article {item['article_number']}"
        assert item["citation"] == expected, f"Invalid citation for article {item['article_number']}"

def test_sane_record_lengths(corpus):
    """Ensure text length is within expected bounds."""
    for item in corpus:
        assert len(item["text_ar"]) < 10000, f"Article {item['article_number']} text is abnormally long"

def test_sorted_and_unique_articles(corpus):
    """Assert article numbers are unique and sorted."""
    numbers = [item["article_number"] for item in corpus]
    assert len(numbers) == len(set(numbers)), "Duplicate article numbers detected"
    assert numbers == sorted(numbers), "Article numbers are not sorted"