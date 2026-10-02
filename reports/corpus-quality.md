# Corpus extraction notes

## Method

The source PDF has English on the left and Arabic on the right. Extracting a whole page interleaves both columns. The ingestion script crops the two columns separately, starts the Arabic pass at the Civil Code heading, reverses Arabic text line by line, and preserves digit order while reversing. Article numbers are normalized to integers. Articles 54–80 are kept as explicit repealed records.

## Validation

- 1,096 records were extracted, numbered from 1 to 1,149.
- Every record has Arabic text, an integer article number, a source page, a citation, and the requested hierarchy fields.
- Articles 54–80 are present and flagged as repealed.
- Twenty records sampled with a fixed random seed were checked for Arabic readability, reversed article numbers, mixed columns, and unreasonable lengths.
- The longest Arabic article is under 1,500 characters.

## Numbers absent from the extracted PDF text

The positional extraction did not yield article records for: 83, 120, 187, 195, 203, 286, 387, 389–417, 428, 450, 473, 584, 658, 676, 690, 714, 756, 760, 765, 811, 888, 901, 1022, 1110, and 1115. These are documented extraction gaps, not assumed repeals. The ingestion script does not invent provisions or silently fill them; the validation test records the exact gap list so a later source-PDF correction is visible.

The PDF's Arabic and English text layers also use inconsistent punctuation and occasional stray one-character extraction artifacts. The parser normalizes common Arabic alef/hamza variants, diacritics, whitespace, and digits, and drops isolated Latin-letter artifacts.
