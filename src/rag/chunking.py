import json
from pathlib import Path
import re
import mlflow
import yaml

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
DIGIT_TRANS = str.maketrans(ARABIC_INDIC_DIGITS, WESTERN_DIGITS)


def parse_arabic_int(val) -> int:
  if isinstance(val, int):
    return val
  val_str = str(val).translate(DIGIT_TRANS)
  digits = re.findall(r"\d+", val_str)
  return int(digits[0]) if digits else 0


def normalize_arabic_text(text: str) -> str:
  if not text:
    return ""
  text = re.sub(r"[\u064B-\u0652]", "", text)
  text = re.sub(r"[إأآا]", "ا", text)
  text = re.sub(r"ى", "ي", text)
  return re.sub(r"\s+", " ", text).strip()


def load_params():
  with open("params.yaml", "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def run_chunking():
  params = load_params()
  input_path = Path(params["raw_data_path"])
  output_path = Path(params["processed_data_path"])

  mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
  mlflow.set_experiment(params["mlflow"]["experiment_name"])

  with mlflow.start_run(run_name="corpus_chunking_stage"):
    if not input_path.exists():
      raise FileNotFoundError(f"Source file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
      raw_articles = json.load(f)

    processed_chunks = []
    repealed_count = 0

    for item in raw_articles:
      art_num = parse_arabic_int(item.get("article_number", 0))
      is_repealed = item.get("is_repealed", False) or (54 <= art_num <= 80)
      if is_repealed:
        repealed_count += 1

      raw_ar = normalize_arabic_text(item.get("text_ar", item.get("text", "")))
      text_en = item.get("text_en", "")

      # Enriched text for better vector embedding retrieval
      enriched_text = f"المادة {art_num}: {raw_ar}" if raw_ar else f"المادة {art_num}"

      chunk = {
          "id": art_num,  # Native integer ID for Qdrant PointStruct
          "chunk_id": f"art_{art_num}",
          "text": enriched_text,  # Rich text fed into Embedding model
          "metadata": {
              "article_number": art_num,
              "book": item.get("book", "General"),
              "chapter": item.get("chapter", ""),
              "section": item.get("section", ""),
              "topic": item.get("topic", ""),
              "text_ar": raw_ar,
              "text_en": text_en,
              "is_repealed": is_repealed,
              "source_page": item.get("source_page", 0),
              "citation": f"Egyptian Civil Code, Article {art_num}",
          },
      }
      processed_chunks.append(chunk)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(processed_chunks, f, ensure_ascii=False, indent=2)

    mlflow.log_params({
        "chunk_strategy": "by_article",
        "raw_input_path": str(input_path),
        "output_path": str(output_path),
    })
    mlflow.log_metrics({
        "total_articles": len(processed_chunks),
        "repealed_articles_count": repealed_count,
    })
    print(
        f"Chunking completed successfully! Total: {len(processed_chunks)} articles."
    )


if __name__ == "__main__":
  run_chunking()