"""Build data/papers_catalogue.csv from the PDFs in data/papers/ (PyMuPDF, first page).

Columns: paper_id, title, authors, year, source, identifier, filename
(pipeline.py needs paper_id / filename / title; the rest is the catalogue the brief requires).

Title and authors are heuristics (largest font on page 1, lines between the title and
"Abstract"), so REVIEW the CSV by hand. Year / source / identifier come from the file name.

Usage:  python -m src.ingest.catalogue            (won't overwrite an existing CSV)
        python -m src.ingest.catalogue --force
"""
import argparse
import csv
import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "data" / "papers"
CATALOGUE = ROOT / "data" / "papers_catalogue.csv"
FIELDS = ["paper_id", "title", "authors", "year", "source", "identifier", "filename"]

_AFFIL = re.compile(r"universit|institut|college|school|laborator|\blab\b|research|department|academy|"
                    r"inc\.|corp|ltd|technolog|google|microsoft|meta|alibaba|tencent|baidu|bytedance|"
                    r"huawei|nvidia|deepmind|openai|china|usa|korea|japan|singapore|@|http|equal contribution|"
                    r"corresponding|project page|github", re.I)


def _source_from_filename(name: str) -> tuple[str, str, str]:
    """(year, source, identifier) inferred from common open-access naming schemes."""
    stem = Path(name).stem
    if m := re.fullmatch(r"(\d{2})(\d{2})\.(\d{4,5})(v\d+)?", stem):          # arXiv YYMM.NNNNN
        return f"20{m.group(1)}", "arXiv", f"arXiv:{m.group(1)}{m.group(2)}.{m.group(3)}{m.group(4) or ''}"
    if m := re.fullmatch(r"(\d{4})\.(acl|emnlp|naacl|eacl|findings)[-\w.]*", stem, re.I):  # ACL Anthology
        return m.group(1), "ACL Anthology", stem
    if m := re.search(r"_(CVPR|ICCV|ECCV|WACV)F?_(\d{4})_paper$", stem):     # CVF Open Access
        return m.group(2), f"CVF Open Access ({m.group(1)})", stem
    if m := re.search(r"AAAI(\d{2})", stem):                                 # AAAI proceedings
        return f"20{m.group(1)}", "AAAI (OJS)", stem
    return "", "", stem


def _title_and_authors(pdf: Path) -> tuple[str, str]:
    with fitz.open(pdf) as doc:
        meta_title = (doc.metadata or {}).get("title", "").strip()
        page = doc[0]
        page_h = page.rect.height
        lines = []   # (y, size, text)
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                if abs(ln.get("dir", (1, 0))[1]) > 0.1:      # rotated text (arXiv side stamp)
                    continue
                text = re.sub(r"\s+", " ", " ".join(s["text"] for s in ln["spans"])).strip()
                if text and not re.match(r"^arXiv:\d", text):
                    lines.append((ln["bbox"][1], max(s["size"] for s in ln["spans"]), text))
    if not lines:
        return meta_title, ""
    lines.sort(key=lambda x: x[0])
    top = [l for l in lines if l[0] < page_h * 0.5]
    biggest = max(l[1] for l in top or lines)
    title_lines = [l for l in (top or lines) if abs(l[1] - biggest) < 0.6]
    title = " ".join(l[2] for l in title_lines)
    if len(title) < 8 and len(meta_title) > 8:
        title = meta_title

    # authors: lines after the title block, before "Abstract", that don't look like affiliations
    last_y = max(l[0] for l in title_lines)
    names: list[str] = []
    for y, size, text in lines:
        if y <= last_y:
            continue
        if re.match(r"^\s*a\s?b\s?s\s?t\s?r\s?a\s?c\s?t\b", text, re.I) or y > page_h * 0.45:
            break
        if _AFFIL.search(text) or len(text) > 200:
            continue
        for part in re.split(r"[,;]|\band\b", re.sub(r"[\d*∗†‡§¶♠♣♥♦⋆]+", " ", text)):
            part = re.sub(r"\s+", " ", part).strip()
            # a person name: 2-5 words, each starting with a capital letter
            if re.fullmatch(r"(?:[A-Z][\w.'\-]*\s){1,4}[A-Z][\w.'\-]*", part) and part not in names:
                names.append(part)
    return _fix_small_caps(re.sub(r"\s+", " ", title).strip()), "; ".join(names[:15])


def _fix_small_caps(title: str) -> str:
    """'F EW MMB ENCH' -> 'FEW MMBENCH' (small-caps fonts extract with split capitals)."""
    return re.sub(r"\b([A-Z]) (?=[A-Z]{2,}\b)", r"\1", title)


def build(force: bool = False) -> Path:
    if CATALOGUE.exists() and not force:
        print(f"{CATALOGUE.name} exists - not overwriting (use --force)")
        return CATALOGUE
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    rows = []
    for i, pdf in enumerate(pdfs, start=1):
        year, source, ident = _source_from_filename(pdf.name)
        try:
            title, authors = _title_and_authors(pdf)
        except Exception as e:  # malformed PDF - keep the row, fill in by hand
            print(f"[warn] {pdf.name}: could not read first page ({type(e).__name__})")
            title, authors = "", ""
        rows.append({"paper_id": f"p{i:02d}", "title": title, "authors": authors, "year": year,
                     "source": source, "identifier": ident, "filename": pdf.name})
    CATALOGUE.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOGUE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} papers to {CATALOGUE.relative_to(ROOT)}")
    return CATALOGUE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    build(ap.parse_args().force)
