import json
import re
import fitz
import requests
import traceback
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy

from jsonschema import validate, ValidationError


DEBUG = True
def debug(*args):
    if DEBUG:
        print(*args)

MODEL = "qwen2.5:7b-instruct"
MAX_DATE_WORDS = 300
CONTEXT_WINDOW = 5

nlp = spacy.load("en_core_web_sm")

ENTITIES = ["MONEY", "PERCENT", "CARDINAL", "QUANTITY"]
KEYWORDS = ["budget", "fiscal", "spend", "alloc"]

money_regex = re.compile(
    r"(?:\$|USD)?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|m|b|bn|thousand|million|billion)?\b",
    re.IGNORECASE,
)

# ----------------- TEXT EXTRACT -----------------
def extract_first_words(pdf_path, max_words):
    doc = fitz.open(pdf_path)
    words = []
    for page in doc:
        page_text = page.get_text("text") or ""
        page_words = re.findall(r"\S+", page_text)
        words.extend(page_words)
        if len(words) >= max_words:
            break
    return words[:max_words]


def _extract_text_from_pdf(pdf_path):
    return "".join(page.get_text("text") + "\n" for page in fitz.open(pdf_path))


# ----------------- MONEY CONTEXT SEARCH -----------------
def _is_money_sentence(sentence: str):
    doc = nlp(sentence)
    return (
        any(ent.label_ in ENTITIES for ent in doc.ents)
        or money_regex.search(sentence)
        or any(k in sentence.lower() for k in KEYWORDS)
    )


def extract_money_contexts(pdf_path):
    text = _extract_text_from_pdf(pdf_path)
    sentences = sent_tokenize(text)

    triggers = [i for i, s in enumerate(sentences) if _is_money_sentence(s)]
    if not triggers:
        return []

    windows = []
    for idx in triggers:
        start = max(0, idx - CONTEXT_WINDOW)
        end = min(len(sentences), idx + CONTEXT_WINDOW + 1)
        windows.append([start, end])

    merged = []
    cur_s, cur_e = windows[0]
    for s, e in windows[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append([cur_s, cur_e])
            cur_s, cur_e = s, e
    merged.append([cur_s, cur_e])

    return [" ".join(sentences[s:e]) for s, e in merged]


# ----------------- JSON SCHEMA -----------------
JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "agency": {"type": "string"},
            "function": {"type": "string"},
            "number": {"type": "string"},
            "number_type": {
                "type": "string",
                "enum": ["Amount", "Percentage"]
            },
            "budget_type": {
                "type": "string",
                "enum": ["Increase", "Decrease", "Total Budget", "Share"]
            }
        },
        "required": ["date", "agency", "function", "number", "number_type", "budget_type"],
        "additionalProperties": False
    }
}




# ----------------- PROMPTS -----------------
SYSTEM_PROMPT = """
You are a STRICT JSON extractor for U.S. intelligence budget documents.
Extract ONLY what is explicitly present in THIS context block.
Accuracy is more important than recall.

════════════════════════
REQUIRED FIELDS (ALWAYS)
════════════════════════
Every JSON object MUST include:
- date  (string)
- agency  (string)
- function  (string, ≤12 words)
- number  (string)
- number_type  ("Amount" or "Percentage")
- budget_type  ("Total Budget", "Increase", "Decrease", or "Share")

════════════════════════
RULES FOR "number" + "number_type"
════════════════════════

1️⃣ Monetary Funding (number_type = "Amount")
- A real currency value appears in the context
- Convert to plain integer US dollars in string form:
    "$52.6B" → "52600000000"
    "$700 million" → "700000000"
- budget_type:
    "Total Budget": clearly a total allocation
    "Increase": explicitly compared and higher
    "Decrease": explicitly compared and lower

2️⃣ Budget Share (number_type = "Percentage")
- A percent value appears in context
- Preserve percent and explicitly state what it is a share of:
    "8% of NIP budget"
- budget_type MUST be "Share"
- DO NOT convert percentages into dollars

════════════════════════
DATE RULES
════════════════════════
- Must be clearly tied to the extracted number in THIS context
- If not visible, use fallback provided by user

════════════════════════
AGENCY RULES
════════════════════════
AGENCY (STRICT RULES)
- The agency MUST be explicitly written near the number
- If the number is written as a “NIP Budget” or “MIP Budget” entry:
  → use “National Intelligence Program (NIP)” or “Military Intelligence Program (MIP)”
- NEVER substitute ODNI for NIP unless the number is directly labeled “ODNI”
- Do not infer a parent agency based on knowledge or associations
- If the agency label is unclear → EXCLUDE the entry

════════════════════════
STRICT PROHIBITIONS
════════════════════════
❌ Do not invent agencies, dates, or numbers
❌ Do not convert percentages into dollars
❌ Do not use earlier contexts for interpretation
❌ Do not output incomplete fields
❌ Do not include narrative text

════════════════════════
OUTPUT FORMAT
════════════════════════
- JSON array only
- No wrapper text
- If zero valid entries → return []
"""




DATE_INFER_SYSTEM_PROMPT = """
Extract the document year.

Return JSON:
{"date": "YYYY" OR "Unknown"}

- Most recent valid year 1900–2050
- If unsure: "Unknown"
"""


# ----------------- OLLAMA CALL -----------------
def call_ollama_api(system_prompt: str, user: str, is_date_inference=False):
    url = "http://localhost:11434/api/generate"
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user}\n<|assistant|>"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": JSON_SCHEMA if not is_date_inference else "json",
        "options": {
            "temperature": 0.0 # consistent output
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=200)
        raw = response.json().get("response", "").strip()
        return raw

    except Exception as e:
        debug(f"❌ Ollama error: {e}")
        return ""


# ----------------- DATE INFERENCE -----------------
def infer_date_from_pdf_first_words(pdf_path):
    words = extract_first_words(pdf_path, MAX_DATE_WORDS)
    if not words:
        return None

    snippet = " ".join(words)
    raw = call_ollama_api(DATE_INFER_SYSTEM_PROMPT, snippet, True)

    try:
        return json.loads(raw).get("date")
    except:
        return None


# ----------------- MAIN PROCESSING -----------------
def convert_contexts(contexts, pdf_path):
    print(f"\n📊 Processing: {pdf_path}")
    fallback_date = infer_date_from_pdf_first_words(pdf_path) or "Unknown"
    fallback_src = "inferred" if fallback_date != "Unknown" else "unknown"
    print(f"🗓 Fallback date: {fallback_date} ({fallback_src})")

    results = []

    def worker(ctx):
        user = f"Fallback Date: {fallback_date}\nContext:\n{ctx}"
        raw = call_ollama_api(SYSTEM_PROMPT, user)

        # Print debug for context + JSON together (pre-parse!), locked in one thread
        if DEBUG:
            print("\n=== CONTEXT BEGIN ===")
            print(ctx[:1000])  # or however much you want
            print("--- JSON RAW BEGIN ---")
            print(raw)
            print("=== DEBUG BLOCK END ===\n")


        try:
            data = json.loads(raw)
        except Exception as e:
            print(e)
            print("⚠️ JSON parse issue — skipping")
            return
        

        if not isinstance(data, list):
            print("⚠️ Expected JSON array — skipping")
            return

        for item in data:
            # Must be a JSON object → otherwise schema violation
            if not isinstance(item, dict):
                print(f"⚠️ Invalid item (not a dict): {item}")
                continue

            try:
                validate(item, JSON_SCHEMA["items"])
            except ValidationError as e:
                print(f"⚠️ Local schema reject: {e.message}")
                continue

            if not isinstance(item, dict):
                continue

            item["context"] = ctx
            results.append(item)

    print(f"⚙️ Running {len(contexts)} contexts...\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, c) for c in contexts]
        for i, _ in enumerate(as_completed(futures), start=1):
            print(f"  ✓ Completed {i}/{len(futures)}")


    out = pdf_path.replace(".pdf", "_extracted.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved → {out}")
    print(f"📈 Valid extracted entries: {len(results)}")
