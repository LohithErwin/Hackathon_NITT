"""Section-aware chunking + figure<->caption<->nearby-text linking."""
import re
from .clean import scan_injection

TARGET_WORDS = 350     # ~450 tokens
MAX_WORDS = 500
REF_HEAD_RE = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$", re.IGNORECASE)
# Publisher watermarks / licence banners that carry no research content
BOILERPLATE_RE = re.compile(
    r"open access version, provided by the computer vision foundation|"
    r"final published version of the proceedings is available on ieee xplore",
    re.IGNORECASE)
SENT_SPLIT =re.compile(r"(?<=[.!?])\s+")


def _words(s: str) -> int:
    return len(s.split())


def _split_long(text: str, max_words: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries."""
    out, cur = [], []
    for s in SENT_SPLIT.split(text):
        if cur and _words(" ".join(cur)) + _words(s) > max_words:
            out.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def _split_table(md: str, max_words: int) -> list[str]:
    """Keep tables whole; if huge, split by rows and repeat the header."""
    if _words(md) <= max_words * 2:
        return [md]
    lines = md.splitlines()
    header, rows = lines[:2], lines[2:]
    parts, cur = [], []
    for r in rows:
        if cur and _words(" ".join(cur + [r])) > max_words:
            parts.append("\n".join(header + cur)); cur = []
        cur.append(r)
    if cur:
        parts.append("\n".join(header + cur))
    return parts


def build_chunks(paper_id: str, title: str, blocks: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    section, in_refs = "Front matter", False
    buf: list[str] = []
    buf_pages: list[int] = []

    def flush():
        nonlocal buf, buf_pages
        if not buf:
            return
        text = " ".join(buf)
        for piece in _split_long(text, MAX_WORDS):
            chunks.append(_mk(piece, "text", buf_pages[0], buf_pages[-1]))
        buf, buf_pages = [], []

    def _mk(text, ctype, p0, p1, caption=""):
        n = len(chunks) + 1
        return {
            "chunk_id": f"{paper_id}::c{n:04d}",
            "paper_id": paper_id,
            "paper_title": title,
            "type": ctype,                       # text | table
            "section": section,
            "page_start": p0, "page_end": p1,
            "caption": caption,
            # Section title is prepended so a chunk is self-describing for embedding.
            "text": f"[{title} | {section}] {text}",
            "raw_text": text,
            "is_references": in_refs,
            "suspicious": scan_injection(text),  # untrusted-content flag
        }

    for b in blocks:
        if b["kind"] in ("text", "heading") and BOILERPLATE_RE.search(b["text"]):
            continue
        if b["kind"] == "heading":
            flush()
            section = b["text"]
            in_refs = bool(REF_HEAD_RE.match(section))
        elif b["kind"] == "text":
            buf.append(b["text"]); buf_pages.append(b["page"])
            if _words(" ".join(buf)) >= TARGET_WORDS:
                flush()
        elif b["kind"] == "table":
            flush()
            cap = b.get("caption", "")
            for part in _split_table(b["text"], MAX_WORDS):
                # Docling's markdown often already embeds the caption; avoid duplicating it
                prefix = "" if cap and cap[:40] in part else cap
                chunks.append(_mk((prefix + "\n" + part).strip(),
                                  "table", b["page"], b["page"], b.get("caption", "")))
    flush()
    return chunks


def build_figures(paper_id: str, title: str, blocks: list[dict]) -> list[dict]:
    """Attach caption + paragraphs that mention the figure ('as shown in Figure 3')."""
    texts = [b for b in blocks if b["kind"] == "text"]
    figs = []
    for i, b in enumerate([b for b in blocks if b["kind"] == "figure"], start=1):
        fid = b.get("fig_id")
        nearby = []
        if fid:
            num = fid.split()[-1]
            pat = re.compile(rf"\b(fig(?:ure)?\.?\s*{num})\b(?!\d)", re.IGNORECASE)
            nearby = [t["text"] for t in texts if pat.search(t["text"])][:3]
        if not nearby:   # fall back to text on the same page
            nearby = [t["text"] for t in texts if t["page"] == b["page"]][:2]
        figs.append({
            "figure_uid": f"{paper_id}::f{i:03d}",
            "paper_id": paper_id,
            "paper_title": title,
            "fig_id": fid,
            "page": b["page"],
            "caption": b.get("caption", ""),
            "caption_missing": not b.get("caption"),
            "nearby_text": nearby,
            "image_path": b["image_path"],
            # this is the text that gets embedded for figure search
            "embed_text": f"[{title}] {fid or 'Figure'} p.{b['page']}: "
                          f"{b.get('caption', '')} {' '.join(nearby)}".strip(),
            "suspicious": scan_injection(b.get("caption", "") + " ".join(nearby)),
        })
    return figs
