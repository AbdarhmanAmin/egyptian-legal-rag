"""Extract the English and Arabic columns of the Civil Code into article records."""

import argparse
import json
import re
from pathlib import Path

import pdfplumber

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
ARABIC_DIGIT_MAP = str.maketrans(ARABIC_DIGITS, WESTERN_DIGITS)
BRACKET_MAP = str.maketrans("()[]{}", ")([]}{")


def restore_arabic_line(line: str) -> str:
    parts = re.split(r"(\d+)", line)
    line = "".join(part if part.isdigit() else part[::-1] for part in reversed(parts))
    return line.translate(BRACKET_MAP)


def normalize_arabic(text: str) -> str:
    text = text.translate(ARABIC_DIGIT_MAP)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[إأآ]", "ا", text).replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def extract_column(
    pdf, box: tuple[float, float, float, float], language: str
) -> dict[int, dict[str, object]]:
    """Read one language column, keeping article text across page breaks."""
    articles: dict[int, dict[str, object]] = {}
    current_number: int | None = None
    started = language != "ar"
    headings = {
        "book": "Egyptian Civil Code",
        "chapter": "",
        "section": "",
        "topic": "",
    }
    for page_number, page in enumerate(pdf.pages, start=1):
        text = page.crop(box).extract_text() or ""
        lines = text.splitlines()
        if language == "ar":
            lines = [restore_arabic_line(line) for line in lines]
            pattern = re.compile(r"^\s*مادة\D{0,8}(\d+)", re.IGNORECASE)
        else:
            pattern = re.compile(r"^\s*Article\s*\(?\s*(\d+)", re.IGNORECASE)

        for line in lines:
            line = (
                normalize_arabic(line)
                if language == "ar"
                else re.sub(r"\s+", " ", line).strip()
            )
            if language == "ar" and not started:
                if "نصوص القانون المدني" in line:
                    started = True
                continue
            heading_patterns = (
                ("book", r"^(?:BOOK|الكتاب|كتاب)\b"),
                ("chapter", r"^(?:CHAPTER|الباب|باب)\b"),
                ("section", r"^(?:SECTION|الفصل|فصل)\b"),
            )
            heading_found = False
            for key, heading_pattern in heading_patterns:
                if re.search(heading_pattern, line, re.IGNORECASE):
                    headings[key] = line
                    heading_found = True
                    break
            if heading_found:
                continue
            topic = (
                re.match(r"^\d+[.)-]\s*(.+)$", line) if current_number is None else None
            )
            if topic:
                headings["topic"] = topic.group(1).strip()
                continue
            if len(line) == 1 and line.isascii() and line.isalpha():
                continue
            match = pattern.search(line)
            if match:
                current_number = int(match.group(1))
                if current_number > 1149:
                    current_number = None
                    continue
                record = articles.setdefault(
                    current_number,
                    {"text": [], "page": page_number, **headings},
                )
                remainder = line[match.end() :].strip(" ()-–:")
                if remainder:
                    record["text"].append(remainder)
            elif current_number is not None and line:
                articles[current_number]["text"].append(line)
    return articles


def parse_pdf_to_corpus(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        width, height = pdf.pages[0].width, pdf.pages[0].height
        english = extract_column(pdf, (0, 0, width * 0.48, height * 0.92), "en")
        arabic = extract_column(pdf, (width * 0.49, 0, width, height * 0.92), "ar")
    numbers = sorted(set(english) | set(arabic) | set(range(54, 81)))
    corpus = []
    for number in numbers:
        ar_record = arabic.get(number, {"text": [], "page": 0})
        en_record = english.get(number, {"text": [], "page": ar_record["page"]})
        text_ar = normalize_arabic(" ".join(ar_record["text"]))
        text_en = re.sub(r"\s+", " ", " ".join(en_record["text"])).strip()
        is_repealed = 54 <= number <= 80
        if is_repealed:
            text_ar = text_ar or "ملغاة بمرسوم رئاسي"
            text_en = text_en or "Repealed by presidential decree"
        if not text_ar:
            continue
        page = ar_record["page"] or en_record["page"]
        corpus.append(
            {
                "article_number": number,
                "book": en_record.get("book")
                or ar_record.get("book")
                or "Egyptian Civil Code",
                "chapter": en_record.get("chapter") or ar_record.get("chapter") or "",
                "section": en_record.get("section") or ar_record.get("section") or "",
                "topic": en_record.get("topic") or ar_record.get("topic") or "",
                "text_ar": text_ar,
                "text_en": text_en,
                "is_repealed": is_repealed,
                "source_page": page,
                "citation": f"Egyptian Civil Code, Article {number}",
            }
        )
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Civil Code article corpus")
    parser.add_argument("--pdf", default="data/raw/القانون المدني المصري.pdf")
    parser.add_argument("--out", default="data/processed/egyptian_civil_code.json")
    args = parser.parse_args()

    corpus = parse_pdf_to_corpus(args.pdf)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    active_articles = [article for article in corpus if not article["is_repealed"]]
    metrics = {
        "article_count": len(corpus),
        "active_article_count": len(active_articles),
        "repealed_article_count": len(corpus) - len(active_articles),
        "arabic_text_coverage": round(
            sum(bool(article["text_ar"]) for article in corpus) / len(corpus), 4
        ) if corpus else 0.0,
        "english_text_coverage": round(
            sum(bool(article["text_en"]) for article in corpus) / len(corpus), 4
        ) if corpus else 0.0,
    }
    metrics_path = Path("reports/corpus_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(corpus)} article records to {output}")


if __name__ == "__main__":
    main()
