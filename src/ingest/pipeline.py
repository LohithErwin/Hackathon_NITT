"""Preprocessing pipeline: PDFs -> data/processed/<paper_id>/{chunks,figures}.jsonl + meta.json

Usage:  python -m src.ingest.pipeline            (all papers in data/papers_catalogue.csv)
        python -m src.ingest.pipeline --only p03
"""
import argparse
import csv
import json
import logging
from pathlib import Path

from .chunk import build_chunks, build_figures
from .parse import parse_pdf, render_pages, validate_pdf

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "data" / "papers"
OUT_DIR = ROOT / "data" / "processed"
CATALOGUE = ROOT / "data" / "papers_catalogue.csv"

log = logging.getLogger("ingest")


def process_paper(meta: dict, pdf_path: Path, out_root: Path = OUT_DIR) -> dict:
    """Process one PDF. Returns a summary dict; raises ValueError with a safe message on bad input."""
    validate_pdf(pdf_path)
    pid = meta["paper_id"]
    out = out_root / pid
    out.mkdir(parents=True, exist_ok=True)

    blocks, report = parse_pdf(pdf_path, out / "figures", pid)
    title = meta.get("title") or pid
    chunks = build_chunks(pid, title, blocks)
    figures = build_figures(pid, title, blocks)
    pages = render_pages(pdf_path, out / "pages")

    for name, rows in (("chunks", chunks), ("figures", figures)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        **meta,
        "n_pages": len(pages),
        "n_chunks": len(chunks),
        "n_text_chunks": sum(c["type"] == "text" for c in chunks),
        "n_table_chunks": sum(c["type"] == "table" for c in chunks),
        "n_figures": len(figures),
        "n_suspicious": sum(c["suspicious"] for c in chunks) + sum(f["suspicious"] for f in figures),
        "parse_report": report,
    }
    (out / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="paper_id to process")
    args = ap.parse_args()

    with open(CATALOGUE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for meta in rows:
        if args.only and meta["paper_id"] != args.only:
            continue
        pdf = PAPERS_DIR / meta["filename"]
        try:
            s = process_paper(meta, pdf)
            log.info("%s: %d pages, %d chunks (%d tables), %d figures, %d flagged, parser=%s",
                     meta["paper_id"], s["n_pages"], s["n_chunks"], s["n_table_chunks"],
                     s["n_figures"], s["n_suspicious"], s["parse_report"]["parser"])
            for w in s["parse_report"]["warnings"]:
                log.warning("%s: %s", meta["paper_id"], w)
        except ValueError as e:
            log.error("%s skipped: %s", meta["paper_id"], e)
        except Exception as e:     # never leak stack traces/paths to users
            log.error("%s failed: %s", meta["paper_id"], type(e).__name__)


if __name__ == "__main__":
    main()
