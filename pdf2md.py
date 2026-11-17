import base64
import re
import sys
from pathlib import Path

import requests
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

# ---------------- config ----------------
OUT_DIR = Path(__file__).parent / "data" / "md"      # default output folder
OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = "qwen2.5vl:3b"          # change to "qwen2.5vl:7b" for better accuracy

FIGURE_PROMPT = """You are an expert at reading figures from research papers.
Explain this figure completely so that someone who cannot see it understands
everything in it. There is no word limit.

Caption: "{caption}"
Text from the paper that refers to this figure:
\"\"\"{context}\"\"\"

Use the caption and the paper text to understand what the figure means, and
the image to describe what it shows. Write these sections (skip any section
that does not apply to this figure):

**Type:** one word or short phrase (e.g. bar chart, line graph, donut chart,
heatmap, diagram, flowchart, model architecture, photo).

**Overview:** what the figure is about and what it is trying to show.

**Components:** title, axes (names, units, ranges), legend entries and what
each color means, each panel / subplot separately. For diagrams, explain each
block and how information flows between them, step by step.

**Data and values:** the numbers you can clearly read, and which item is
highest / lowest.

**Trends and comparisons:** patterns, differences between groups, outliers.

**Key takeaway:** the conclusion the authors draw - it must agree with the
caption and the paper text above.

Rules:
- Only describe what is visible or stated in the caption / paper text.
- Never invent axes, legends, labels or numbers. If something is not clearly
  readable, write "not readable".
- No preamble."""

TABLE_PROMPT = """You are an expert at reading tables from research papers.
Explain this table completely so that someone who cannot see it understands
everything in it. There is no word limit.

Caption: "{caption}"
The table, already extracted as markdown (use THESE numbers - they are exact):
{table_md}

The image is the same table; use it only to check the layout (merged cells,
bold / underlined values). Write these sections (skip any section that does
not apply):

**Overview:** what the table is about and what it compares.

**Structure:** the column headers and what each measures; the rows / row
groups and what each represents.

**Contents:** go through the table row by row and report the values exactly.
Mention bold / underlined / starred entries and what they mean.

**Best and worst results:** ONLY if the table reports results / scores. For
each metric, which row is best and which is worst, with numbers. If the table
is a list of settings or configurations, skip this section.

**Comparisons and trends:** how the proposed method compares with others (by
how much), patterns, ablation effects. Skip for configuration tables.

**Key takeaway:** the main conclusion.

Rules:
- Every number you write must come from the table above. Never invent values.
- Read each value from the correct column - do not shift columns.
- A "-" means the value was not reported.
- No preamble."""

FORMULA_PROMPT = """Transcribe the mathematical equation in this image into LaTeX.
Output ONLY the LaTeX code - no $$ signs, no explanation, no markdown."""
# ----------------------------------------

use_vlm = True   # switched off automatically if Ollama is not running
FIG_RE = re.compile(r"^\s*Fig(?:ure)?\.?\s*(\d+)", re.IGNORECASE)


def ask_vlm(img_path: Path, prompt: str, temperature: float = 0.1) -> str:
    """Send one image + prompt to the vision model. Returns '' on failure."""
    global use_vlm
    if not use_vlm:
        return ""
    payload = {
        "model": VLM_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(img_path.read_bytes()).decode()],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": -1, "num_ctx": 8192},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=900)
        r.raise_for_status()
        return r.json()["response"].strip()
    except requests.RequestException as e:
        print(f"[warn] Ollama failed ({e}); continuing without the vision model")
        use_vlm = False
        return ""


def as_quote(text: str) -> str:
    """Put text inside a markdown quote block."""
    if not text:
        return "> (description skipped - Ollama not running)"
    return "\n".join(f"> {line}" for line in text.splitlines())


_converter = None


def get_converter():
    """Build the (slow to create) Docling converter once and reuse it."""
    global _converter
    if _converter is not None:
        return _converter
    # ---- Docling settings (accuracy first) ----
    opts = PdfPipelineOptions()
    opts.images_scale = 2.0                                  # sharper crops
    opts.generate_page_images = True                         # needed to crop equations
    opts.generate_picture_images = True
    opts.generate_table_images = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE   # better tables
    opts.do_formula_enrichment = True                        # equations -> LaTeX
    _converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter


def convert(pdf, out_dir=OUT_DIR) -> Path:
    """Convert one PDF to markdown. Returns the path of the .md file."""
    global use_vlm
    use_vlm = True
    pdf, out_dir = Path(pdf), Path(out_dir)
    assets = out_dir / "assets" / pdf.stem          # one assets folder per paper
    assets.mkdir(parents=True, exist_ok=True)
    rel = f"assets/{pdf.stem}"                      # link used inside the markdown

    print(f"Converting {pdf.name} (formula + accurate table mode, this takes a while) ...")
    doc = get_converter().convert(str(pdf)).document

    items = [it for it, _ in doc.iterate_items()]
    all_texts = [it.text for it in items
                 if it.label.value == "text" and getattr(it, "text", "")]

    # captions already attached to a figure/table are written next to it, not twice
    skip = {c.cref for it in list(doc.pictures) + list(doc.tables) for c in it.captions}

    md = []
    n_fig = n_tab = n_eq = 0

    for i, item in enumerate(items):
        label = item.label.value
        if label in ("page_header", "page_footer") or item.self_ref in skip:
            continue

        # ---------- figures / charts / graphs ----------
        if label == "picture":
            img = item.get_image(doc)
            if img is None or img.width < 50 or img.height < 50:   # skip tiny icons
                continue
            n_fig += 1
            name = f"figure_{n_fig:02d}.png"
            img.save(assets / name)

            caption = item.caption_text(doc).strip()
            if not caption:   # Docling missed it: use the next "Figure N: ..." text
                for nxt in items[i + 1:i + 10]:
                    t = getattr(nxt, "text", "")
                    if FIG_RE.match(t):
                        caption = t.strip()
                        skip.add(nxt.self_ref)
                        break
            caption = caption or f"Image {n_fig} (no caption)"

            # paragraphs that mention this figure ("as shown in Figure 3")
            m = FIG_RE.match(caption)
            num = m.group(1) if m else str(n_fig)
            pat = re.compile(rf"\bFig(?:ure)?\.?\s*{num}\b", re.IGNORECASE)
            mentions = [t[:800] for t in all_texts if pat.search(t) and t != caption]
            context = "\n".join(mentions[:3]) or "(none)"

            print(f"  figure {n_fig}: describing ...")
            desc = ask_vlm(assets / name,
                           FIGURE_PROMPT.format(caption=caption, context=context))
            md += [f"![Figure {n_fig}]({rel}/{name})",
                   f"**{caption}**  [open image]({rel}/{name})",
                   "> **Detailed description:**", as_quote(desc), ""]

        # ---------- tables ----------
        elif label == "table":
            n_tab += 1
            name = f"table_{n_tab:02d}.png"
            caption = item.caption_text(doc).strip()
            table_md = item.export_to_markdown(doc=doc)
            # Docling's table markdown often contains the caption already
            if caption and caption[:40] not in table_md:
                md.append(f"**{caption}**\n")
            md.append(table_md)

            img = item.get_image(doc)
            if img is not None:
                img.save(assets / name)
                print(f"  table {n_tab}: describing ...")
                desc = ask_vlm(assets / name, TABLE_PROMPT.format(
                    caption=caption or f"Unnumbered table {n_tab}", table_md=table_md))
                md += ["", f"[open table image]({rel}/{name})",
                       "> **Detailed description:**", as_quote(desc)]
            md.append("")

        # ---------- equations ----------
        elif label == "formula":
            latex = item.text.strip()
            if not latex:   # Docling could not read it -> ask the vision model
                img = item.get_image(doc)
                if img is not None:
                    n_eq += 1
                    name = f"equation_{n_eq:02d}.png"
                    img.save(assets / name)
                    latex = ask_vlm(assets / name, FORMULA_PROMPT, temperature=0.0)
                    latex = latex.strip().strip("$").strip()
                    if not latex:
                        md.append(f"[equation image]({rel}/{name})\n")
                        continue
            if latex:
                md.append(f"$$\n{latex}\n$$\n")

        # ---------- text ----------
        elif label == "title":
            md.append(f"# {item.text}\n")
        elif label == "section_header":
            md.append(f"{'#' * (getattr(item, 'level', 1) + 1)} {item.text}\n")
        elif label == "list_item":
            md.append(f"- {item.text}")
        elif label == "code":
            md.append(f"```\n{item.text}\n```\n")
        elif getattr(item, "text", "").strip():
            md.append(item.text + "\n")

    out_md = out_dir / f"{pdf.stem}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nDone: {out_md}")
    print(f"Saved {n_fig} figure(s), {n_tab} table(s), {n_eq} equation image(s) in {assets}")
    return out_md


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python pdf2md.py paper.pdf")
    convert(sys.argv[1])