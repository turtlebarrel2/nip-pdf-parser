import os
import subprocess
import requests
import hashlib
from pdf_to_context import extract_money_contexts
from context_to_json import convert_contexts

TEMP_DIR = "./tmp_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)


def _hash_file(path: str) -> str:
    """Generate hash to compare original vs cached OCR input source"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_pdf(url: str) -> str:
    filename = os.path.join(TEMP_DIR, url.split("/")[-1] or "document.pdf")

    # If downloaded file already exists, reuse it
    if os.path.exists(filename):
        print(f"📁 Using cached download: {filename}")
        return filename

    print(f"⬇️ Downloading from web: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"📄 Saved: {filename}")
    return filename


def get_ocr_pdf_path(pdf_path: str) -> str:
    base, ext = os.path.splitext(pdf_path)
    return f"{base}_ocr.pdf"


def ensure_ocr_pdf(pdf_path: str) -> str:
    """Use OCR'd PDF if cached and matching, otherwise regenerate."""

    ocr_path = get_ocr_pdf_path(pdf_path)
    hash_path = f"{ocr_path}.hash"

    # Check if cached OCR exists and hash matches original input
    if os.path.exists(ocr_path) and os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            cached_hash = f.read().strip()

        current_hash = _hash_file(pdf_path)

        if cached_hash == current_hash:
            print(f"👍 Using cached OCR: {ocr_path}")
            return ocr_path

        print(f"⚠️ Source changed — regenerating OCR PDF")

    # Perform OCR if no valid cached version exists
    print(f"🔧 Running OCR on: {pdf_path}")
    try:
        subprocess.run([
            "ocrmypdf",
            "--force-ocr",
            "--deskew",
            "--clean",
            "--rotate-pages",
            "--invalidate-digital-signatures",
            "--output-type", "pdf",
            "-O", "0",
            pdf_path,
            ocr_path
        ], check=True)

        # Save hash so reuse is safe
        with open(hash_path, "w") as f:
            f.write(_hash_file(pdf_path))

        print(f"📄 OCR completed: {ocr_path}")
        return ocr_path

    except Exception as e:
        print(f"❌ OCR FAILED ({e}) — using original instead\n")
        return pdf_path


PDF_URLS = [
    "https://ia601305.us.archive.org/28/items/TheUSIntelligenceCommunity/Doc%2034-NIP%20Budget%20%282012%29.pdf",
    "https://www.govinfo.gov/content/pkg/CRPT-118hrpt162/pdf/CRPT-118hrpt162.pdf",
    "https://www.cbo.gov/system/files/2023-11/hr3932.pdf",
    "https://www.congress.gov/crs_external_products/R/PDF/R44381/R44381.11.pdf"
]


if __name__ == "__main__":
    for url in PDF_URLS:
        print(f"\n=== Processing New PDF ===\n{url}")

        pdf_path = download_pdf(url)
        processed_pdf = pdf_path#ensure_ocr_pdf(pdf_path)

        contexts = extract_money_contexts(processed_pdf)
        convert_contexts(contexts, processed_pdf)
