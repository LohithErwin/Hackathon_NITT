
import hashlib
import re
from pathlib import Path

import rag

# headings that belong to each summary part (checked in order, case-insensitive)
SUMMARY_PARTS = [
    ("Introduction", ("introduction", "background", "motivation")),
    ("Method", ("method", "approach", "architecture", "framework", "model", "proposed")),
    ("Results", ("result", "experiment", "evaluation", "benchmark")),
    ("Conclusion", ("conclusion", "discussion", "summary")),
]
SNIPPET_CHARS = 400
ABSTRACT_CHARS = 1500

_converter = None


def validate_pdf(pdf_path):
    """Raise ValueError unless pdf_path is an acceptable PDF (same rules as library uploads)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise ValueError("File not found.")
    rag.validate_pdf(pdf_path.name, pdf_path.read_bytes())


def _fast_converter():
    """Docling without OCR / formula / picture models: text + tables only, much faster."""
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        opts.do_formula_enrichment = False
        opts.generate_picture_images = False
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter


def convert(pdf_path):
    """PDF -> markdown text (sanitised). Re-uses the library's markdown when possible."""
    pdf_path = Path(pdf_path)
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    papers, _, _ = rag.load_index()
    for pid, p in papers.items():
        md_file = rag.MD_DIR / f"{pid}.md"
        if p.get("hash") == digest and md_file.exists():
            return rag.scan(md_file.read_text(encoding="utf-8"))[0]

    doc = _fast_converter().convert(str(pdf_path)).document
    return rag.scan(doc.export_to_markdown())[0]


def _clean(text):
    """Drop images, '[open image]' links, VLM '> ...' descriptions and markdown tables."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)|\[open (table )?image\]\([^)]*\)|\[equation image\]\([^)]*\)",
                  "", text)
    lines = [l for l in text.splitlines()
             if not l.lstrip().startswith(">") and not l.lstrip().startswith("|")]
    text = "\n".join(lines).replace("**", "").replace("<!-- image -->", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def sections(md):
    """Markdown -> ordered list of (heading, text). Text before the first heading is 'Start'."""
    out, heading, buf = [], "Start", []
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.+)$", line)
        if m:
            out.append((heading, _clean("\n".join(buf))))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    out.append((heading, _clean("\n".join(buf))))
    return [(h, t) for h, t in out if t or h != "Start"]


def get_title(secs, fallback):
    """The first heading is the paper title."""
    for heading, _ in secs:
        if heading != "Start":
            return heading
    return fallback


def _strip_number(heading):
    return re.sub(r"^\s*(\d+(\.\d+)*|[IVX]+)\.?\s*", "", heading).strip()


def _find(secs, name):
    """Text of the first section whose heading contains `name`. '' if none.
    For 'abstract' it also handles papers whose abstract has no heading."""
    for heading, text in secs:
        if name.lower() in heading.lower() and text:
            return text
    if name.lower() == "abstract" and secs:
        # 'Abstract. Vision-language models ...' inside the title section, or the
        # first long paragraph after the author list
        first = secs[1][1] if secs[0][0] == "Start" and len(secs) > 1 else secs[0][1]
        for para in re.split(r"\n\s*\n", first):
            para = para.strip()
            if re.match(r"(?i)abstract\b", para):
                return re.sub(r"(?i)^abstract[.:\s-]*", "", para)
        long = [p.strip() for p in re.split(r"\n\s*\n", first) if len(p.strip()) > 400]
        if long:
            return long[0]
    return ""


def get_authors(secs):
    """Best-effort author line: first short paragraph under the title."""
    for heading, text in secs[:2]:
        if heading == "Start":
            continue
        para = re.split(r"\n\s*\n", text.strip())[0] if text.strip() else ""
        if 0 < len(para) < 400 and "," in para:
            # split on commas, 'and', and affiliation digits glued to the next name ('Huy1 Trang')
            names = [re.sub(r"[\s\d∗*†‡§,]+$", "", n).strip(" ∗*†‡§")
                     for n in re.split(r",|\band\b|(?<=[\d∗*†‡§])\s+(?=[A-Z])", para)]
            return [n for n in names if 2 < len(n) < 40]
        return []
    return []


def extractive_summary(secs):
    """'Abstract: ...\\n\\nIntroduction: ...\\n\\nMethod: ...' built from the section texts."""
    abstract = " ".join(_find(secs, "abstract").split())[:ABSTRACT_CHARS]
    parts = [f"Abstract: {abstract}"] if abstract else []
    used = set()
    for label, keys in SUMMARY_PARTS:
        for i, (heading, text) in enumerate(secs):
            name = _strip_number(heading).lower()
            if i in used or not text or "abstract" in name:
                continue
            if any(k in name for k in keys):
                used.add(i)
                parts.append(f"{label}: {' '.join(text.split())[:SNIPPET_CHARS]}")
                break
    if not parts and secs:                       # nothing recognisable: use the beginning
        parts.append(" ".join(secs[0][1].split())[:ABSTRACT_CHARS])
    return "\n\n".join(parts)
