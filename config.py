from pathlib import Path

ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "data" / "papers"
STORE_DIR = ROOT / "data" / "stores"
for d in (PAPERS_DIR, STORE_DIR):
    d.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_PDF_MB = 50
OCR = False   # True only for scanned PDFs; arXiv papers have real text, and OCR is slow