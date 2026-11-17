import sys
import time
from pathlib import Path
import parse
from store import PaperDB, ChunkDB

def process(pdf_path: Path):
    t0 = time.time()
    parse.validate_pdf(pdf_path)
    paper_id = pdf_path.stem
    doc = parse.convert(pdf_path)

    # DB1: paper-level record (title, authors, extractive summary)
    secs = parse.sections(doc)
    title = parse.get_title(secs, paper_id)
    authors = parse.get_authors(secs)
    summary = parse.extractive_summary(secs)
    PaperDB().add(paper_id, title, authors, summary)

    # DB2: full detail, every text chunk and table, with page + section
    base = {"paper_id": paper_id, "paper_title": title, "authors": ", ".join(authors)}
    chunks = [{"id": f"{paper_id}_front", "text": parse.front_matter(doc),
               "meta": {**base, "type": "front_matter", "page": 1, "section": "front"}}]
    for c in parse.text_chunks(doc, paper_id):
        c["meta"].update(base)
        chunks.append(c)
    for t in parse.tables(doc):
        chunks.append({"id": f"{paper_id}_tbl_{t['idx']}",
                       "text": f"{t['caption']}\n\n{t['markdown']}",
                       "meta": {**base, "type": "table", "page": t["page"], "section": "table"}})
    ChunkDB().add(chunks)

    print(f"Done: {title} | {len(chunks)} chunks | {time.time() - t0:.0f}s")

if __name__ == "__main__":
    for p in sys.argv[1:]:                      # accepts several PDFs in one run
        process(Path(p))