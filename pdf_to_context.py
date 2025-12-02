import fitz
import pdfplumber
import re
import spacy
from nltk.tokenize import sent_tokenize

# 🔧 Toggle this to enable/disable table extraction logic
USE_TABLE_EXTRACTION = True

CONTEXT_WINDOW = 5
nlp = spacy.load("en_core_web_sm")

ENTITIES = ["MONEY", "PERCENT", "CARDINAL", "QUANTITY"]
KEYWORDS = ["budget", "fiscal", "spend", "alloc"]

money_regex = re.compile(
    r"(?:\$|USD)?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|m|b|bn|thousand|million|billion)?\b",
    re.IGNORECASE,
)


def _is_money_sentence(sentence: str):
    doc = nlp(sentence)
    return (
        any(ent.label_ in ENTITIES for ent in doc.ents)
        or money_regex.search(sentence)
        or any(k in sentence.lower() for k in KEYWORDS)
    )



# -------- MAIN EXTRACTION FUNCTION --------

import pandas as pd

def extract_money_contexts(pdf_path):
    contexts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            sentences = sent_tokenize(text)

            # ===== TABLE EXTRACTION (Pandas-Based Row Sentences) =====
            if USE_TABLE_EXTRACTION:
                tables = page.extract_tables()
                print(tables)
                for table in tables:
                    if not table:
                        continue

                    try:
                        df = pd.DataFrame(table)
                        df = df.fillna("")

                        # Clean whitespace in every cell
                        df = df.applymap(lambda v: " ".join(v.split()) if isinstance(v, str) else v)

                        # Iterate rows as “label: value; label: value” format
                        for r_index, row in df.iterrows():
                            parts = []
                            for c_index, cell in enumerate(row):
                                header = df.columns[c_index]
                                header_clean = " ".join(header.split()) if isinstance(header, str) else f"Col{c_index+1}"
                                cell_clean = cell if cell else "[EMPTY]"
                                parts.append(f"{header_clean}: {cell_clean}")

                            sentence = (
                                f"(Page {page_index+1}) Table Row {r_index+1}: "
                                + "; ".join(parts)
                                + "."
                            )
                            contexts.append(sentence)

                    except Exception:
                        # Worst fallback: flatten cells
                        flat = [(" ".join(cell.split()) if cell else "[EMPTY]") for row in table for cell in row]
                        contexts.append(f"(Page {page_index+1}) " + "; ".join(flat) + ".")


            # ===== TEXT EXTRACTION =====
            triggers = [i for i, s in enumerate(sentences) if _is_money_sentence(s)]
            if not triggers:
                continue

            windows = []
            for idx in triggers:
                start = max(0, idx - CONTEXT_WINDOW)
                end = min(len(sentences), idx + CONTEXT_WINDOW + 1)
                windows.append([start, end])

            # Merge overlapping windows
            merged = []
            cur_s, cur_e = windows[0]
            for s, e in windows[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    merged.append([cur_s, cur_e])
                    cur_s, cur_e = s, e
            merged.append([cur_s, cur_e])

            # Add merged blocks
            for s, e in merged:
                block = " ".join(sentences[s:e]).strip()
                if block:
                    contexts.append(f"(Page {page_index+1})\n{block}")

    return contexts

