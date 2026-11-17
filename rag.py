
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

# ---------------- config ----------------
MD_DIR = Path(__file__).parent / "preprocessing/output"          # where pdf2md.py writes .md files
INDEX_FILE = Path(__file__).parent / "rag_index.json"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"            # any text model you have, e.g. "llama3.2:3b"
CHUNK_SIZE = 1200                   # max characters in one text chunk
# ----------------------------------------

# patterns to find "Figure 4", "Fig. 4", "Table 2", "Eq. 3", "Equation (3)"
FIG_RE = re.compile(r"\bfig(?:ure)?\.?\s*(\d+)", re.IGNORECASE)
TAB_RE = re.compile(r"\btable\.?\s*(\d+)", re.IGNORECASE)
EQ_RE = re.compile(r"\beq(?:uation)?\.?\s*\(?(\d+)", re.IGNORECASE)


# =====================================================================
# Step 0: talking to Ollama
# =====================================================================

def embed(texts):
    """Turn a list of texts into a numpy array of vectors (one row per text)."""
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBED_MODEL, "input": texts}, timeout=600)
    r.raise_for_status()
    vectors = np.array(r.json()["embeddings"], dtype=np.float32)
    # normalise so that a dot product = cosine similarity
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def ask_llm(prompt):
    """Send a prompt to the text model and return its answer."""
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }, timeout=900)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# =====================================================================
# Step 1: PDF -> Markdown (reuses your pdf2md.py)
# =====================================================================

def convert_pdfs(pdf_folder):
    """Run pdf2md.py on every PDF in a folder (skips ones already converted)."""
    script = Path(__file__).parent / "pdf2md.py"
    for pdf in sorted(Path(pdf_folder).glob("*.pdf")):
        if (MD_DIR / f"{pdf.stem}.md").exists():
            print(f"skip {pdf.name} (already converted)")
            continue
        print(f"converting {pdf.name} ...")
        subprocess.run([sys.executable, str(script), str(pdf)], check=True)


# =====================================================================
# Step 2: Markdown -> blocks with metadata
# =====================================================================

def section_number(section):
    """'3.2 Results' -> '3.2'   ,  'Abstract' -> '' """
    m = re.match(r"\s*(\d+(?:\.\d+)*)", section)
    return m.group(1) if m else ""


def quote_to_text(lines):
    """Turn '> ...' quote lines (the VLM description) back into plain text."""
    out = [l.strip().lstrip(">").strip() for l in lines]
    return "\n".join(l for l in out if l and l != "**Detailed description:**")


def parse_markdown(md):
    """
    Read the markdown written by pdf2md.py and cut it into blocks.
    pdf2md.py separates every element with a blank line, so we read
    one paragraph at a time and decide what it is:
        '# ...'          -> heading (changes the current section)
        '![Figure ...'   -> figure  (caption + VLM description)
        '| ... |'        -> table   (caption + markdown table + description)
        '$$'             -> equation
        anything else    -> normal text
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip()]
    blocks, section = [], "Start"
    n_fig = n_tab = n_eq = 0

    for para in paragraphs:
        lines = para.splitlines()
        first = lines[0].strip()

        # ---- heading ----
        if first.startswith("#") and len(lines) == 1:
            section = first.lstrip("#").strip()
            continue

        # ---- figure ----
        if first.startswith("![Figure"):
            n_fig += 1
            caption = lines[1] if len(lines) > 1 else ""
            caption = re.sub(r"\[open image\]\([^)]*\)", "", caption).replace("**", "").strip()
            m = re.match(r"\s*fig(?:ure)?\.?\s*(\d+)", caption, re.IGNORECASE)
            image = re.search(r"\(([^)]+)\)", first)
            blocks.append({"type": "figure", "section": section,
                           "number": int(m.group(1)) if m else n_fig,
                           "caption": caption,
                           "image": image.group(1) if image else "",
                           "text": quote_to_text(lines[2:])})
            continue

        # ---- table description (comes right after its table) ----
        if first.startswith("[open table image]"):
            if blocks and blocks[-1]["type"] == "table":
                blocks[-1]["text"] += "\n\nDescription:\n" + quote_to_text(lines[1:])
            continue

        # ---- table ----
        if any(l.strip().startswith("|") for l in lines):
            n_tab += 1
            table_lines = [l for l in lines if l.strip().startswith("|")]
            caption = " ".join(l.strip() for l in lines if not l.strip().startswith("|"))
            # the caption may be the paragraph just before the table
            if not caption and blocks and blocks[-1]["type"] == "text" \
                    and TAB_RE.match(blocks[-1]["text"].replace("**", "")):
                caption = blocks.pop()["text"]
            caption = caption.replace("**", "").strip()
            m = re.match(r"\s*table\.?\s*(\d+)", caption, re.IGNORECASE)
            blocks.append({"type": "table", "section": section,
                           "number": int(m.group(1)) if m else n_tab,
                           "caption": caption,
                           "text": "\n".join(table_lines)})
            continue

        # ---- equation ----
        if first.startswith("$$") or first.startswith("[equation image]"):
            n_eq += 1
            latex = para.replace("$$", "").strip()
            if latex.startswith("[equation image]"):
                latex = "(equation could not be read)"
            blocks.append({"type": "equation", "section": section,
                           "number": n_eq, "caption": "", "text": latex})
            continue

        # ---- normal text ----
        blocks.append({"type": "text", "section": section, "text": para})

    return blocks


def make_chunks(blocks):
    """
    Figures, tables and equations each become ONE chunk.
    Normal text paragraphs of the same section are joined
    until the chunk reaches CHUNK_SIZE characters.
    """
    chunks, buffer = [], []

    def flush():
        if buffer:
            text = "\n\n".join(b["text"] for b in buffer)
            chunks.append({"type": "text", "section": buffer[0]["section"],
                           "number": None, "caption": "", "text": text})
            buffer.clear()

    for b in blocks:
        if b["type"] != "text":
            flush()
            chunks.append(b)
            continue
        size = sum(len(x["text"]) for x in buffer)
        if buffer and (b["section"] != buffer[0]["section"] or size + len(b["text"]) > CHUNK_SIZE):
            flush()
        buffer.append(b)
    flush()

    for c in chunks:
        c["section_no"] = section_number(c["section"])
        # which figures / tables this chunk talks about ("as shown in Figure 4")
        c["mentions"] = {"figure": sorted({int(n) for n in FIG_RE.findall(c["text"])}),
                         "table": sorted({int(n) for n in TAB_RE.findall(c["text"])})}
    return chunks


def label(c):
    """Human-readable source label, e.g. [Paper 2: Title | 4 Results | Figure 4]"""
    what = f" | {c['type'].capitalize()} {c['number']}" if c["number"] else ""
    return f"[Paper {c['paper_no']}: {c['title']} | {c['section']}{what}]"


def embed_text(c):
    """The text we turn into a vector: metadata + content, so both are searchable."""
    head = f"Paper: {c['title']}\nSection: {c['section']}\n"
    if c["number"]:
        head += f"{c['type'].capitalize()} {c['number']}: {c['caption']}\n"
    return head + c["text"]


# =====================================================================
# Step 3: build / load the index
# =====================================================================

def get_title(md, fallback):
    m = re.search(r"^# (.+)$", md, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def build_index():
    md_files = sorted(MD_DIR.glob("*.md"))
    if not md_files:
        sys.exit(f"No .md files in {MD_DIR}. Run pdf2md.py (or 'python rag.py convert') first.")

    papers, all_chunks = {}, []
    for no, md_file in enumerate(md_files, 1):
        md = md_file.read_text(encoding="utf-8")
        pid, title = md_file.stem, get_title(md, md_file.stem)
        chunks = make_chunks(parse_markdown(md))
        for c in chunks:
            c.update({"paper_no": no, "paper_id": pid, "title": title})
        all_chunks += chunks

        # paper-level metadata
        clean = re.sub(r"!\[[^\]]*\]\([^)]*\)|\[open (table )?image\]\([^)]*\)", "", md)
        papers[pid] = {
            "number": no, "title": title, "text": clean,
            "figures": [c["caption"] or f"Figure {c['number']}" for c in chunks if c["type"] == "figure"],
            "tables": [c["caption"] or f"Table {c['number']}" for c in chunks if c["type"] == "table"],
            "sections": list(dict.fromkeys(c["section"] for c in chunks)),
        }
        print(f"Paper {no}: {title}  ({len(papers[pid]['figures'])} figures, "
              f"{len(papers[pid]['tables'])} tables, {len(chunks)} chunks)")

    print(f"Embedding {len(all_chunks)} chunks ...")
    vectors = []
    for i in range(0, len(all_chunks), 32):              # 32 chunks per request
        vectors.append(embed([embed_text(c) for c in all_chunks[i:i + 32]]))
    for c, v in zip(all_chunks, np.vstack(vectors)):
        c["vector"] = v.tolist()

    INDEX_FILE.write_text(json.dumps({"papers": papers, "chunks": all_chunks}), encoding="utf-8")
    print(f"Saved index to {INDEX_FILE}")


def load_index():
    if not INDEX_FILE.exists():
        sys.exit("No index yet. Run:  python rag.py build")
    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    vectors = np.array([c["vector"] for c in data["chunks"]], dtype=np.float32)
    return data["papers"], data["chunks"], vectors


# =====================================================================
# Step 4: retrieval = metadata filter + similarity search
# =====================================================================

def matches(c, where):
    """True if chunk c has all the metadata in `where`, e.g. {"paper_no": 2, "type": "figure"}."""
    for key, value in where.items():
        if key == "section_no":       # "section 3" also matches 3.1, 3.2 ...
            if not (c["section_no"] == value or c["section_no"].startswith(value + ".")):
                return False
        elif c.get(key) != value:
            return False
    return True


def search(question, chunks, vectors, top_k=5, where=None):
    """Most similar chunks to the question, only among chunks that match `where`."""
    allowed = [i for i, c in enumerate(chunks) if matches(c, where or {})]
    if not allowed:
        return []
    q = embed([question])[0]
    scores = vectors[allowed] @ q                        # cosine similarity
    best = np.argsort(-scores)[:top_k]
    return [(float(scores[j]), chunks[allowed[j]]) for j in best]


def format_context(chunks_list):
    """Join chunks into one block of text for the LLM, each with its metadata label."""
    parts = []
    for c in chunks_list:
        cap = f"Caption: {c['caption']}\n" if c["caption"] else ""
        parts.append(f"{label(c)}\n{cap}{c['text']}")
    return "\n\n---\n\n".join(parts)


# =====================================================================
# Step 5: understand the question ("paper 2 figure 4")
# =====================================================================

def parse_question(question, papers):
    """Pull metadata out of the question. Returns a dict for matches()."""
    where = {}
    q = question.lower()

    m = re.search(r"\bpaper\s*#?\s*(\d+)", q)              # "paper 2"
    if m:
        where["paper_no"] = int(m.group(1))
    else:                                                    # or the title / file name
        for pid, p in papers.items():
            if pid.lower() in q or p["title"].lower() in q:
                where["paper_no"] = p["number"]
                break

    for kind, rx in (("figure", FIG_RE), ("table", TAB_RE), ("equation", EQ_RE)):
        m = rx.search(q)
        if m:
            where["type"], where["number"] = kind, int(m.group(1))
            break

    m = re.search(r"\bsection\s*(\d+(?:\.\d+)*)", q)         # "section 3.2"
    if m:
        where["section_no"] = m.group(1)
    return where


# =====================================================================
# Function 1: details about one paper
# =====================================================================

def paper_details(paper_id, papers, chunks, vectors):
    paper = papers[paper_id]
    beginning = paper["text"][:4000]                        # title, authors, abstract, intro
    extra = search("methods, experiments, datasets, results and conclusion",
                   chunks, vectors, top_k=4, where={"paper_id": paper_id, "type": "text"})

    answer = ask_llm(f"""You are given parts of a research paper. Using ONLY this text,
write a clear summary with these sections:

**Title:**
**Authors:** (write "not found" if missing)
**Problem:** what problem the paper solves
**Method:** the proposed approach, in simple words
**Datasets / Experiments:**
**Key results:** with numbers if given
**Conclusion:**
**Limitations / future work:** (write "not mentioned" if missing)

Paper beginning:
\"\"\"{beginning}\"\"\"

Other relevant parts:
\"\"\"{format_context([c for _, c in extra])}\"\"\"
""")
    # metadata straight from the index (no LLM needed)
    meta = [f"\n\n**Paper number:** {paper['number']}   **File:** {paper_id}",
            "**Sections:** " + ", ".join(paper["sections"]),
            f"**Figures ({len(paper['figures'])}):**"] + [f"- {f}" for f in paper["figures"]] + \
           [f"**Tables ({len(paper['tables'])}):**"] + [f"- {t}" for t in paper["tables"]]
    return answer + "\n".join(meta)


# =====================================================================
# Function 2: find the papers relevant to a question / topic
# =====================================================================

def find_relevant_papers(question, papers, chunks, vectors, top_n=3):
    results = search(question, chunks, vectors, top_k=30)

    best = {}                       # score of a paper = score of its best chunk
    for score, c in results:
        if c["paper_id"] not in best or score > best[c["paper_id"]][0]:
            best[c["paper_id"]] = (score, c)
    ranked = sorted(best.values(), key=lambda x: -x[0])[:top_n]

    lines = []
    for rank, (score, c) in enumerate(ranked, 1):
        why = ask_llm(f"""Question: "{question}"
Text from the paper "{c['title']}":
\"\"\"{c['text']}\"\"\"
In 2-3 sentences, explain how this paper is relevant to the question.
If it is not relevant, say so.""")
        lines.append(f"{rank}. Paper {c['paper_no']}: {c['title']}  (score {score:.2f})\n"
                     f"   Best match: {label(c)}\n   {why}")
    return "\n\n".join(lines) if lines else "No relevant papers found."


# =====================================================================
# Function 3: compare several papers
# =====================================================================

def compare_papers(paper_ids, papers, chunks, vectors, aspect=""):
    topic = aspect or "problem, method, datasets, results, strengths and limitations"
    sections = []
    for pid in paper_ids:
        found = search(topic, chunks, vectors, top_k=4, where={"paper_id": pid})
        sections.append(f"### PAPER {papers[pid]['number']}: {papers[pid]['title']}\n"
                        f"{papers[pid]['text'][:1500]}\n\n"
                        f"{format_context([c for _, c in found])}")

    return ask_llm(f"""Compare the following research papers.
Focus on: {topic}

Use ONLY the text below. Write:
1. A markdown table with one row per paper and columns:
   Paper | Problem | Method | Datasets | Main results | Limitations
2. **Similarities:**
3. **Differences:**
4. **Which paper is better for what:** (only if the text supports it)
Write "not mentioned" when something is missing. Do not invent numbers.

{chr(10).join(sections)}
""")


# =====================================================================
# Function 4: ask anything (uses the metadata in the question)
# =====================================================================

def ask(question, papers, chunks, vectors):
    where = parse_question(question, papers)

    if "type" in where:
        # exact item, e.g. paper 2 + figure 4
        context = [c for c in chunks if matches(c, where)]
        if not context:
            return not_found_message(where, papers)
        # + text paragraphs of the same paper that talk about this figure/table
        for c in chunks:
            if c["type"] == "text" and c["paper_no"] in {x["paper_no"] for x in context} \
                    and where["number"] in c["mentions"].get(where["type"], []):
                context.append(c)
        context = context[:6]
    else:
        # no figure/table asked -> similarity search inside the filter (paper / section)
        context = [c for _, c in search(question, chunks, vectors, top_k=6, where=where)]
        if not context:
            return not_found_message(where, papers)

    answer = ask_llm(f"""Answer the question using ONLY the paper excerpts below.
Each excerpt starts with [Paper number: title | section | item].
If the answer is not in the excerpts, say "Not found in the papers."
Mention which paper / figure / table your answer comes from.

Question: {question}

Excerpts:
{format_context(context)}
""")
    sources = "\n".join(f"- {label(c)}" for c in context)
    return f"{answer}\n\nSources:\n{sources}"


def not_found_message(where, papers):
    """Explain what is missing and list what IS available."""
    asked = ", ".join(f"{k}={v}" for k, v in where.items())
    msg = f"Nothing found for ({asked})."
    target = [p for p in papers.values() if p["number"] == where.get("paper_no")]
    if not target and "paper_no" in where:
        return msg + f" There are only {len(papers)} papers."
    for p in target or papers.values():
        items = p["figures"] if where.get("type") == "figure" else \
                p["tables"] if where.get("type") == "table" else p["sections"]
        msg += f"\nPaper {p['number']} has: " + ("; ".join(items) or "none")
    return msg


# =====================================================================
# Simple menu
# =====================================================================

def show_papers(papers):
    ids = list(papers)
    for pid in ids:
        p = papers[pid]
        print(f"  {p['number']}. {p['title']}  ({len(p['figures'])} fig, {len(p['tables'])} tab)")
    return ids


def menu():
    papers, chunks, vectors = load_index()
    print(f"Loaded {len(papers)} papers, {len(chunks)} chunks.")
    while True:
        print("\n1) Paper details   2) Find relevant papers   3) Compare papers")
        print("4) Ask a question (e.g. 'explain paper 2 figure 4')   q) Quit")
        choice = input("> ").strip().lower()
        try:
            if choice == "1":
                ids = show_papers(papers)
                pid = ids[int(input("Paper number: ")) - 1]
                print("\n" + paper_details(pid, papers, chunks, vectors))
            elif choice == "2":
                q = input("Topic or question: ")
                print("\n" + find_relevant_papers(q, papers, chunks, vectors))
            elif choice == "3":
                ids = show_papers(papers)
                picks = input("Paper numbers separated by commas (e.g. 1,3): ")
                chosen = [ids[int(x) - 1] for x in picks.split(",")]
                aspect = input("Compare on what? (Enter = everything): ").strip()
                print("\n" + compare_papers(chosen, papers, chunks, vectors, aspect))
            elif choice == "4":
                show_papers(papers)
                q = input("Question: ")
                print("\n" + ask(q, papers, chunks, vectors))
            elif choice == "q":
                break
        except (ValueError, IndexError):
            print("Invalid choice, try again.")
        except requests.RequestException as e:
            print(f"Ollama error: {e}  (is 'ollama serve' running?)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        convert_pdfs(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        menu()