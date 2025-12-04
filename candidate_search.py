import os
import subprocess
import requests
import hashlib
import json
import glob

from pdf_to_context import extract_money_contexts_from_mineru
from context_to_json import convert_contexts
from pdf_urls import PDF_URLS

import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TEMP_DIR = "./tmp_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)

FAILED_LOG = "failed_downloads.txt"


def _hashed_filename(url: str) -> str:
    """
    Create a unique local filename for each URL to avoid collisions/overwrites.
    """
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    tail = url.split("/")[-1] or "document.pdf"
    return os.path.join(TEMP_DIR, f"{digest}__{tail}")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)"
]

DEFAULT_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
}

def download_pdf(url: str) -> str | None:
    filename = _hashed_filename(url)

    # Already downloaded
    if os.path.exists(filename):
        print(f"📁 Using cached download: {filename}")
        return filename

    print(f"⬇️ Downloading with spoofed headers: {url}")

    session = requests.Session()

    # Retry strategy for common failures
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[403, 408, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )

    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    headers = DEFAULT_HEADERS.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)

    try:
        r = session.get(url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()

        # Validate PDF signature early to avoid writing HTML errors
        chunk = next(r.iter_content(chunk_size=2048))
        if b"%PDF" not in chunk[:10]:
            raise ValueError("Server returned non-PDF content")

        with open(filename, "wb") as f:
            f.write(chunk)  # write first validated chunk
            for chunk in r.iter_content(chunk_size=2048):
                if chunk:
                    f.write(chunk)

        print(f"📄 Saved: {filename}")
        return filename

    except Exception as e:
        print(f"❌ Download failed: {url} — {e}")
        with open(FAILED_LOG, "a") as f:
            f.write(url + "\n")

        # Avoid caching corrupt partial file
        if os.path.exists(filename):
            os.remove(filename)

        return None


def _file_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()[:32]  # Unique identity for this exact PDF


def run_mineru(pdf_path: str) -> str | None:
    if not pdf_path:
        return None

    base = os.path.splitext(pdf_path)[0]
    output_dir = base + "_mineru"

    # Unique ID based on file content — not filename
    pdf_hash = _file_hash(pdf_path)

    # Cached output check with content validation
    existing_jsons = glob.glob(os.path.join(output_dir, "**", "*_content_list.json"), recursive=True)
    for existing in existing_jsons:
        try:
            with open(existing, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and data and data[0].get("pdf_sha256") == pdf_hash:
                print(f"👍 Using valid cached MinerU JSON: {existing}")
                return existing
        except Exception:
            pass  # ignore malformed or legacy JSON

    print(f"🧠 Running MinerU fresh on: {pdf_path}")
    try:
        subprocess.run(["mineru", "-p", pdf_path, "-o", output_dir], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ MinerU failed on: {pdf_path}")
        return None

    # Find new JSON from MinerU instead of guessing folder name
    json_candidates = glob.glob(os.path.join(output_dir, "**", "*_content_list.json"), recursive=True)
    if not json_candidates:
        print("❌ MinerU output missing!")
        return None

    # Take the most recently modified JSON
    json_path = max(json_candidates, key=os.path.getmtime)

    # Inject validation metadata into MinerU-generated JSON
    try:
        with open(json_path, "r+") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    item["pdf_sha256"] = pdf_hash
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    except Exception as e:
        print(f"⚠️ Unable to embed PDF hash into JSON ({e})")

    print(f"📄 MinerU JSON identified: {json_path}")
    return json_path

if __name__ == "__main__":
    print("\n🚀 Starting Batch Processing\n")

    for url in PDF_URLS:
        print(f"\n=== Processing New PDF ===\n{url}")

        pdf_path = download_pdf(url)
        if not pdf_path:
            continue  # Skip failed downloads

        json_path = run_mineru(pdf_path)
        if not json_path:
            continue  # Skip failed MinerU runs

        contexts = extract_money_contexts_from_mineru(json_path)
        # 🔐 Pass URL so it’s stored in each JSON item
        convert_contexts(contexts, pdf_path, source_url=url)

    print("\n🎯 Batch complete!")
    print(f"⚠️ Any failed URLs were logged in {FAILED_LOG}")