import re
import json
import argparse
from pathlib import Path
import pdfplumber

def fix_arabic_reversal(text: str) -> str:
    """Fix character-reversed Arabic text extracted by pdfplumber in visual LTR order."""
    if not text:
        return ""
    # Check if Arabic characters exist
    if re.search(r'[\u0600-\u06FF]', text):
        # Reverse string to restore original RTL order
        reversed_text = text[::-1]
        # Swap reversed bracket characters
        brackets_map = str.maketrans("()[]{}", ")(][}{")
        return reversed_text.translate(brackets_map)
    return text

def normalize_arabic_digits(text: str) -> str:
    """Convert Eastern Arabic-Indic numerals (٠-٩) to Western digits (0-9)."""
    if not text:
        return ""
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"
    return text.translate(str.maketrans(arabic_digits, western_digits))

def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic characters, remove diacritics and collapse whitespaces."""
    if not text:
        return ""
    # First fix text direction if reversed
    text = fix_arabic_reversal(text)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    text = re.sub(r'\u0649', '\u064A', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_cell(cell: str) -> str:
    if not cell:
        return ""
    return re.sub(r'\s+', ' ', cell).strip()

def is_strictly_repealed(text_ar: str, text_en: str, art_num: int) -> bool:
    """Check if article is officially repealed without triggering false positives."""
    if 54 <= art_num <= 80:
        return True
    # Check if the text explicitly starts with repeal status or is short repeal statement
    ar_match = bool(re.search(r'^\s*\(?\s*(ملغاة|مُلغاة|ملغاه)\b', text_ar))
    en_match = bool(re.search(r'^\s*\(?\s*\[?\s*repealed\b', text_en, re.IGNORECASE))
    return ar_match or en_match

def parse_pdf_to_corpus(pdf_path: str) -> list[dict]:
    corpus = []

    current_book = "General Provisions / أحكام عامة"
    current_chapter = "Preliminary Chapter / باب تمهيدي"
    current_section = "Laws and their Applications / القانون وتطبيقه"
    current_topic = "General Rules / أحكام عامة"

    art_en_pattern = re.compile(r'Article\s*(\d+)', re.IGNORECASE)
    art_ar_pattern = re.compile(r'مادة[^\d0-9]*([\d0-9]+)', re.IGNORECASE)
    repealed_pattern = re.compile(r'المواد[^\d0-9]*([\d0-9]+)[^\d0-9]*إلى[^\d0-9]*([\d0-9]+)', re.IGNORECASE)

    curr_art_num = None
    curr_ar_lines = []
    curr_en_lines = []
    curr_is_repealed = False
    curr_source_page = 1

    def flush_current_article():
        nonlocal curr_art_num, curr_ar_lines, curr_en_lines, curr_is_repealed, curr_source_page
        if curr_art_num is None:
            return

        text_ar = normalize_arabic_text(" ".join(curr_ar_lines))
        text_en = clean_cell(" ".join(curr_en_lines))

        is_rep = is_strictly_repealed(text_ar, text_en, curr_art_num) or curr_is_repealed

        if not text_ar:
            if text_en and re.search(r'[\u0600-\u06FF]', text_en):
                text_ar = normalize_arabic_text(text_en)
            elif text_en:
                text_ar = text_en
            else:
                text_ar = f"مادة {curr_art_num}"

        corpus.append({
            "article_number": curr_art_num,
            "book": current_book,
            "chapter": current_chapter,
            "section": current_section,
            "topic": current_topic,
            "text_ar": text_ar if not is_rep else f"Repealed: {text_ar or 'ملغاة'}",
            "text_en": text_en if not is_rep else f"Repealed: {text_en or 'Repealed'}",
            "is_repealed": is_rep,
            "source_page": curr_source_page,
            "citation": f"Egyptian Civil Code, Article {curr_art_num}"
        })

        curr_art_num = None
        curr_ar_lines = []
        curr_en_lines = []
        curr_is_repealed = False

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            rows_to_process = []

            if tables:
                for table in tables:
                    for row in table:
                        if not row:
                            continue
                        if len(row) >= 2:
                            cell_en = row[0] or ""
                            cell_ar = row[1] or ""
                        elif len(row) == 1:
                            cell_en = row[0] or ""
                            cell_ar = row[0] or ""
                        else:
                            continue
                        rows_to_process.append((cell_en, cell_ar))
            else:
                raw_text = page.extract_text() or ""
                lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
                for line in lines:
                    if '|' in line:
                        parts = line.split('|')
                        rows_to_process.append((parts[0], parts[1]))
                    else:
                        rows_to_process.append((line, line))

            for cell_en_raw, cell_ar_raw in rows_to_process:
                cell_ar_norm = normalize_arabic_digits(cell_ar_raw)
                cell_en_clean = clean_cell(cell_en_raw)
                cell_ar_clean = clean_cell(cell_ar_norm)

                if not cell_ar_clean and re.search(r'[\u0600-\u06FF]', cell_en_clean):
                    cell_ar_clean = cell_en_clean

                ar_check_text = fix_arabic_reversal(cell_ar_clean)

                if "BOOK" in cell_en_clean.upper() or "الكتاب" in ar_check_text:
                    if curr_ar_lines:
                        flush_current_article()
                    current_book = f"{cell_en_clean} / {ar_check_text}".strip(" /")
                    continue
                elif "CHAPTER" in cell_en_clean.upper() or "الباب" in ar_check_text:
                    if curr_ar_lines:
                        flush_current_article()
                    current_chapter = f"{cell_en_clean} / {ar_check_text}".strip(" /")
                    continue
                elif "SECTION" in cell_en_clean.upper() or "الفصل" in ar_check_text:
                    if curr_ar_lines:
                        flush_current_article()
                    current_section = f"{cell_en_clean} / {ar_check_text}".strip(" /")
                    continue
                elif any(cell_en_clean.startswith(p) for p in ["1.", "2.", "3.", "4.", "5."]) or "أولاً" in ar_check_text or "ثانياً" in ar_check_text:
                    if curr_ar_lines:
                        flush_current_article()
                    current_topic = f"{cell_en_clean} / {ar_check_text}".strip(" /")

                rep_match = repealed_pattern.search(cell_ar_clean) or repealed_pattern.search(cell_en_clean)
                if rep_match:
                    flush_current_article()
                    start_art = int(rep_match.group(1))
                    end_art = int(rep_match.group(2))
                    for art_num in range(start_art, end_art + 1):
                        corpus.append({
                            "article_number": art_num,
                            "book": current_book,
                            "chapter": current_chapter,
                            "section": current_section,
                            "topic": current_topic,
                            "text_ar": "Repealed: ملغاة",
                            "text_en": "Repealed: Repealed",
                            "is_repealed": True,
                            "source_page": page_num,
                            "citation": f"Egyptian Civil Code, Article {art_num}"
                        })
                    continue

                match_en = art_en_pattern.search(cell_en_clean)
                match_ar = art_ar_pattern.search(cell_ar_clean) or art_ar_pattern.search(ar_check_text)

                if match_en or (match_ar and "المواد" not in cell_ar_clean and "المواد" not in ar_check_text):
                    art_num = int(match_en.group(1)) if match_en else int(match_ar.group(1))

                    if art_num != curr_art_num:
                        flush_current_article()
                        curr_art_num = art_num
                        curr_source_page = page_num

                    curr_ar_lines.append(cell_ar_clean)
                    curr_en_lines.append(cell_en_clean)
                else:
                    if curr_art_num is not None:
                        if cell_ar_clean:
                            curr_ar_lines.append(cell_ar_clean)
                        if cell_en_clean:
                            curr_en_lines.append(cell_en_clean)

    flush_current_article()

    unique_corpus = {}
    for item in corpus:
        art_num = item["article_number"]
        if art_num in unique_corpus:
            existing = unique_corpus[art_num]
            if len(item["text_ar"]) > len(existing["text_ar"]):
                existing["text_ar"] = item["text_ar"]
            if len(item["text_en"]) > len(existing["text_en"]):
                existing["text_en"] = item["text_en"]
            if item["is_repealed"] or (54 <= art_num <= 80):
                existing["is_repealed"] = True
        else:
            if 54 <= art_num <= 80:
                item["is_repealed"] = True
            unique_corpus[art_num] = item

    return [unique_corpus[k] for k in sorted(unique_corpus.keys())]

def main():
    parser = argparse.ArgumentParser(description="Convert Egyptian Civil Code PDF to JSON")
    parser.add_argument("--pdf", type=str, default="data/raw/القانون المدني المصري.pdf", help="Input PDF path")
    parser.add_argument("--out", type=str, default="data/processed/egyptian_civil_code.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Processing PDF file: {args.pdf}")
    corpus = parse_pdf_to_corpus(args.pdf)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Extraction finished successfully! Total articles processed: {len(corpus)}")
    print(f"Corpus saved at: {out_path.resolve()}")

if __name__ == "__main__":
    main()