import json
import re
from nltk.tokenize import sent_tokenize
import spacy
from bs4 import BeautifulSoup

CONTEXT_WINDOW = 5
MAX_CONTEXT_CHARS = 3000
MAX_TABLE_ROWS = 25

nlp = spacy.load("en_core_web_sm")
ENTITIES = ["MONEY", "PERCENT", "CARDINAL", "QUANTITY"]
KEYWORDS = ["budget", "fiscal", "spend", "alloc"]

money_regex = re.compile(
    r"(?:\$|USD)?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|m|b|bn|thousand|million|billion)?\b",
    re.IGNORECASE,
)


def _is_money_sentence(sentence: str):
    doc = nlp(sentence)
    s = sentence.lower()
    return (
        any(ent.label_ in ENTITIES for ent in doc.ents)
        or money_regex.search(sentence)
        or any(k in s for k in KEYWORDS)
    )


def _parse_table_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    table_rows = soup.find_all("tr")
    if not table_rows:
        return rows

    header_cells = table_rows[0].find_all(["td", "th"])
    headers = [h.get_text(" ", strip=True) or f"Col{i+1}" for i, h in enumerate(header_cells)]

    for tr in table_rows[1:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        row_pairs = []
        for i, cell in enumerate(cells):
            header = headers[i] if i < len(headers) else f"Col{i+1}"
            row_pairs.append(f"{header}: {cell}")
        rows.append("; ".join(row_pairs))
    return rows


def _chunk_rows(rows, max_chars=MAX_CONTEXT_CHARS, max_rows=MAX_TABLE_ROWS):
    chunk, chunk_len = [], 0

    for row in rows:
        row_len = len(row)

        # Split huge single row into independent safe chunks
        if row_len > max_chars:
            if chunk:
                yield chunk
                chunk, chunk_len = [], 0
            for i in range(0, row_len, max_chars):
                yield [row[i:i + max_chars]]
            continue

        if chunk and (chunk_len + row_len > max_chars or len(chunk) >= max_rows):
            yield chunk
            chunk, chunk_len = [], 0

        chunk.append(row)
        chunk_len += row_len

    if chunk:
        yield chunk


def _add_safely(contexts, page_idx, text):
    """Split *any* final block into ≤3000 char chunks."""
    page_prefix = f"(Page {page_idx+1})\n"
    for i in range(0, len(text), MAX_CONTEXT_CHARS - len(page_prefix)):
        contexts.append(page_prefix + text[i:i + (MAX_CONTEXT_CHARS - len(page_prefix))])


def extract_money_contexts_from_mineru(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages_text = {}
    pages_tables = {}

    # Group content by page
    for obj in data:
        page = obj.get("page_idx")
        if page is None:
            continue
        if obj.get("type") == "text":
            pages_text.setdefault(page, []).append(obj.get("text", "").strip())
        elif obj.get("type") == "table":
            pages_tables.setdefault(page, [])
            pages_tables[page].extend(_parse_table_html(obj.get("table_body", "")))

    contexts = []
    all_pages = sorted(set(pages_text.keys()) | set(pages_tables.keys()))

    # Process text pages
    for page_idx in all_pages:
        if page_idx in pages_text:

            sentences = []
            for block in pages_text[page_idx]:
                sentences.extend(sent_tokenize(block))

            triggers = [i for i, s in enumerate(sentences) if _is_money_sentence(s)]

            if triggers:
                windows = [(max(0, i - CONTEXT_WINDOW),
                            min(len(sentences), i + CONTEXT_WINDOW + 1))
                           for i in triggers]

                # Merge windows but do not trust any merge length
                merged = []
                cur_s, cur_e = windows[0]
                for s, e in windows[1:]:
                    temp_e = max(cur_e, e)
                    cur_e = temp_e
                merged.append((cur_s, cur_e))

                for s, e in merged:
                    text_block = " ".join(sentences[s:e])

                    # Attach tables if present
                    if page_idx in pages_tables:
                        chunks = list(_chunk_rows(pages_tables[page_idx]))
                        for i, chunk in enumerate(chunks, 1):
                            table_block = f"\nTABLE (Chunk {i}/{len(chunks)}):\n" + " | ".join(chunk)
                            _add_safely(contexts, page_idx, text_block + table_block)
                    else:
                        _add_safely(contexts, page_idx, text_block)

                pages_tables.pop(page_idx, None)

    # Leftover pages: TABLE ONLY
    for page_idx in sorted(pages_tables.keys()):
        chunks = list(_chunk_rows(pages_tables[page_idx]))
        for i, chunk in enumerate(chunks, 1):
            table_block = f"TABLE (Chunk {i}/{len(chunks)}):\n" + " | ".join(chunk)
            _add_safely(contexts, page_idx, table_block)

    for c in contexts:
        print(len(c))
    return contexts
