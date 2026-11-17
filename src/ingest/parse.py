"""PDF parsing: Docling (layout/tables/figures) with PyMuPDF fallback + page renders.

Output of parse_pdf() is a list of ordered "blocks":
  {"kind": "heading"|"text"|"table"|"figure", "page": int, "text": str, ...}
"""
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from .clean import clean_text

log = logging.getLogger(__name__)

MIN_CHARS_PER_PAGE = 50          # below this a page is treated as scanned/empty
SKIP_LABELS = {"page_header", "page_footer", "footnote"}
FIG_RE = re.compile(r"^\s*(fig(?:ure)?\.?\s*(\d+))", re.IGNORECASE)


def validate_pdf(path: Path, max_mb: int = 50) -> None:
    """Raise ValueError (with a safe message) if the file is not an acceptable PDF."""
    if path.suffix.lower() != ".pdf":
        raise ValueError("Unsupported file type: only .pdf is accepted.")
    if not path.is_file():
        raise ValueError("File not found.")
    if path.stat().st_size > max_mb * 1024 * 1024:
        raise ValueError(f"File too large (limit {max_mb} MB).")
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("File is not a valid PDF.")
    try:
        with fitz.open(path) as d:
            if d.page_count == 0:
                raise ValueError("PDF has no pages.")
    except ValueError:
        raise
    except Exception:
        raise ValueError("PDF is malformed or encrypted.")


def render_pages(pdf_path: Path, out_dir: Path, dpi: int = 110) -> dict[int, str]:
    """Render every page to PNG (used by the VLM for tables/figures Docling missed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with fitz.open(pdf_path) as d:
        for i, page in enumerate(d, start=1):
            p = out_dir / f"page_{i:03d}.png"
            page.get_pixmap(dpi=dpi).save(p)
            paths[i] = str(p)
    return paths


def find_scanned_pages(pdf_path: Path) -> list[int]:
    """Pages with (almost) no extractable text -> candidates for OCR."""
    bad = []
    with fitz.open(pdf_path) as d:
        for i, page in enumerate(d, start=1):
            if len(page.get_text().strip()) < MIN_CHARS_PER_PAGE:
                bad.append(i)
    return bad


def _ocr_page_text(pdf_path: Path, page_no: int) -> str:
    """Best-effort OCR via PyMuPDF+Tesseract; returns '' if Tesseract is unavailable."""
    try:
        with fitz.open(pdf_path) as d:
            tp = d[page_no - 1].get_textpage_ocr(full=True)
            return d[page_no - 1].get_text(textpage=tp)
    except Exception:
        return ""


def _parse_with_pymupdf(pdf_path: Path, fig_dir: Path, paper_id: str) -> list[dict]:
    """Fallback parser: block-level text (sorted for multi-column) + embedded images."""
    blocks: list[dict] = []
    fig_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as d:
        for pno, page in enumerate(d, start=1):
            for b in page.get_text("blocks", sort=True):
                if b[6] != 0:      # skip image blocks
                    continue
                t = clean_text(b[4])
                if len(t) > 2:
                    blocks.append({"kind": "text", "page": pno, "text": t, "label": "text"})
            for n, img in enumerate(page.get_images(full=True)):
                try:
                    pix = fitz.Pixmap(d, img[0])
                    if pix.width < 120 or pix.height < 120:
                        continue
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    p = fig_dir / f"{paper_id}_p{pno}_img{n}.png"
                    pix.save(p)
                    blocks.append({"kind": "figure", "page": pno, "text": "", "caption": "",
                                   "fig_id": None, "image_path": str(p)})
                except Exception:
                    continue
    return blocks


def _parse_with_docling(pdf_path: Path, fig_dir: Path, paper_id: str) -> list[dict]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False                  # OCR is handled per-page by us (keeps 4GB GPU/CPU light)
    opts.do_table_structure = True
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = conv.convert(str(pdf_path)).document

    fig_dir.mkdir(parents=True, exist_ok=True)
    blocks: list[dict] = []
    fig_count = 0
    for item, _level in doc.iterate_items():
        label = getattr(item.label, "value", str(item.label))
        page = item.prov[0].page_no if getattr(item, "prov", None) else 0
        if label in SKIP_LABELS:
            continue
        if label == "picture":
            cap = clean_text(item.caption_text(doc)) if hasattr(item, "caption_text") else ""
            img_path = None
            try:
                img = item.get_image(doc)
                if img is not None and img.width >= 120 and img.height >= 120:
                    fig_count += 1
                    p = fig_dir / f"{paper_id}_p{page}_fig{fig_count}.png"
                    img.save(p)
                    img_path = str(p)
            except Exception:
                pass
            if img_path:
                m = FIG_RE.match(cap)
                blocks.append({"kind": "figure", "page": page, "text": "", "caption": cap,
                               "fig_id": f"Figure {m.group(2)}" if m else None,
                               "image_path": img_path})
        elif label == "table":
            try:
                md = item.export_to_markdown(doc=doc)
            except Exception:
                md = ""
            cap = clean_text(item.caption_text(doc)) if hasattr(item, "caption_text") else ""
            if md.strip():
                blocks.append({"kind": "table", "page": page, "text": md, "caption": cap})
        elif label == "caption":
            continue                      # attached to its figure/table above
        else:
            t = clean_text(getattr(item, "text", ""))
            if not t:
                continue
            kind = "heading" if label in ("section_header", "title") else "text"
            blocks.append({"kind": kind, "page": page, "text": t, "label": label})
    return blocks


def parse_pdf(pdf_path: Path, fig_dir: Path, paper_id: str) -> tuple[list[dict], dict]:
    """Parse one PDF. Returns (blocks, report). Never raises on extraction problems."""
    report = {"parser": "docling", "scanned_pages": [], "ocr_pages": [], "warnings": []}
    try:
        blocks = _parse_with_docling(pdf_path, fig_dir, paper_id)
        if not any(b["kind"] in ("text", "heading") for b in blocks):
            raise RuntimeError("docling returned no text")
    except Exception as e:
        log.warning("Docling failed for %s (%s); using PyMuPDF fallback", paper_id, type(e).__name__)
        report["parser"] = "pymupdf"
        report["warnings"].append("Docling failed; used PyMuPDF fallback (no table structure).")
        blocks = _parse_with_pymupdf(pdf_path, fig_dir, paper_id)

    # scanned / empty pages -> try OCR, otherwise record the gap
    pages_with_text = {b["page"] for b in blocks if b["kind"] in ("text", "heading")}
    for pno in find_scanned_pages(pdf_path):
        report["scanned_pages"].append(pno)
        if pno in pages_with_text:
            continue
        ocr = clean_text(_ocr_page_text(pdf_path, pno))
        if len(ocr) >= MIN_CHARS_PER_PAGE:
            blocks.append({"kind": "text", "page": pno, "text": ocr, "label": "ocr"})
            report["ocr_pages"].append(pno)
        else:
            report["warnings"].append(f"Page {pno}: no extractable text (OCR unavailable or empty).")
    return blocks, report
