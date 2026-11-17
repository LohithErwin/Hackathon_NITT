import re
from pathlib import Path
import numpy as np
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from config import MAX_PDF_MB, OCR

# ---------- validation + conversion ----------
def validate_pdf(path: Path):
    if path.suffix.lower() != ".pdf":
        raise ValueError("Unsupported file type")
    if path.stat().st_size > MAX_PDF_MB * 1024 * 1024:
        raise ValueError("File too large")
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("Not a valid PDF")

_converter = None
def _get_converter():
    global _converter
    if _converter is None:                      # build once, reuse across papers
        opts = PdfPipelineOptions()
        opts.do_ocr = OCR
        opts.do_table_structure = True          # needed for table text
        opts.generate_picture_images = False    # no image rendering = much faster
        opts.generate_page_images = False
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter

def convert(path: Path):
    return _get_converter().convert(str(path)).document

# ---------- DB2 inputs ----------
def text_chunks(doc, paper_id: str):
    out = []
    for i, ch in enumerate(HybridChunker().chunk(doc)):
        pages = sorted({p.page_no for it in ch.meta.doc_items for p in it.prov})
        out.append({
            "id": f"{paper_id}_txt_{i}",
            "text": ch.text,
            "meta": {"paper_id": paper_id, "type": "text",
                     "page": pages[0] if pages else -1,
                     "section": " > ".join(ch.meta.headings or [])},
        })
    return out

def tables(doc):
    out = []
    for i, t in enumerate(doc.tables):
        out.append({"idx": i,
                    "markdown": t.export_to_markdown(doc),
                    "caption": t.caption_text(doc) or "",
                    "page": t.prov[0].page_no if t.prov else -1})
    return out

def front_matter(doc, n=2000):
    return doc.export_to_markdown()[:n]

# ---------- DB1 inputs: title, authors, extractive summary ----------
def sections(doc):
    secs, cur = [], {"name": "front", "label": "front", "lines": []}
    for t in doc.texts:
        lab = getattr(t.label, "value", t.label)
        if lab in ("title", "section_header"):
            secs.append(cur)
            cur = {"name": t.text.strip(), "label": lab, "lines": []}
        elif lab in ("text", "paragraph", "list_item"):
            cur["lines"].append(t.text.strip())
    secs.append(cur)
    if not any(s["label"] == "title" for s in secs):    # Docling often labels it section_header
        for s in secs:
            if s["label"] == "section_header":
                s["label"] = "title"
                break
    return secs

def get_title(secs, fallback):
    for s in secs:
        if s["label"] == "title":
            return s["name"]
    return fallback

def get_authors(secs):
    for s in secs:
        if s["label"] == "title":
            return [l for l in s["lines"][:8] if len(l) < 300]
    return []

def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if 40 <= len(p.strip()) <= 400]

def _top_sentences(text, k):
    from store import embed
    sents = _sentences(text)
    if len(sents) <= k:
        return sents
    E = embed(sents)
    scores = E @ E.mean(axis=0)
    keep = sorted(np.argsort(-scores)[:k])
    return [sents[i] for i in keep]

def _find(secs, *keywords):
    return " ".join(" ".join(s["lines"]) for s in secs
                    if any(k in s["name"].lower() for k in keywords))

def extractive_summary(secs):
    abstract = _find(secs, "abstract")
    if not abstract:
        abstract = " ".join(" ".join(s["lines"]) for s in secs[:3])[:1500]
    parts = [f"Abstract: {abstract.strip()}"]
    for label, keys, k in [("Introduction", ("introduction",), 3),
                           ("Method", ("method", "approach", "architecture"), 3),
                           ("Results", ("result", "experiment", "evaluation"), 3),
                           ("Conclusion", ("conclusion", "discussion"), 3)]:
        sents = _top_sentences(_find(secs, *keys), k)
        if sents:
            parts.append(f"{label}: " + " ".join(sents))
    return "\n\n".join(parts)