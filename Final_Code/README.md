# ResearchIQ

ResearchIQ lets you ask questions about a library of research papers, including their figures, tables and equations. It can summarise a paper, find the papers that match a topic, and compare papers. You can also upload a new paper and compare it with the library without adding it.

Everything runs locally: [Docling](https://github.com/docling-project/docling) parses the PDFs, and [Ollama](https://ollama.com) runs the vision, embedding and text models.

## Contents

- [Architecture](#architecture)
- [Files](#files)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## Architecture

The app combines two parts that share one paper library (`data/index.json`):

1. **RAG pipeline** (`pdf2md.py` → `rag.py` → `app.py`): answers questions using chunks of each paper.
2. **Paper Explorer** (`store.py` / `parse.py` / `compare.py` → `app_amal.py`): searches and compares whole papers.

```
                       ┌──────────────────────── INGEST (once per paper) ───────────────────────┐
 PDF ─► validate_pdf ─► pdf2md.convert ─► markdown ─► scan ─► parse_markdown ─► make_chunks ─► embed ─► data/index.json
        (security)      Docling +                    (prompt-   text / figure /   (nomic-embed-text)
                        Qwen2.5-VL describes         injection   table / equation
                        figures & tables             filter)     blocks + metadata
                       └─────────────────────────────────────────────────────────────────────────┘
                                                        │
                ┌───────────────────────────────────────┴────────────────────────────────┐
                ▼                                                                         ▼
     CHUNK-LEVEL (rag.py)                                                    PAPER-LEVEL (store.py)
     metadata filter + cosine search                                         1 vector per paper (mean of its chunks)
     ask · paper_details · find_relevant_papers · compare_papers             FAISS IndexFlatIP / NumPy fallback
     answers written by llama3 (Ollama)                                       PaperDB.search ─► search.py, Topic search tab
                                                                              compare.py ─► Upload & compare tab
                                                                                 parse.py sections + extractive summary
                                                                                 TL;DR: fine-tuned SciTLDR (BART), else llama3
                                                                                 slots: metric / model scale / datasets
```

### Ingest pipeline (`rag.add_pdf`)

| Stage | Function | What it does |
|---|---|---|
| 1. Validate | `rag.validate_pdf` | Checks the `.pdf` extension, the size limit (50 MB) and the `%PDF-` / `%%EOF` markers. |
| 2. Convert | `pdf2md.convert` | Docling converts the PDF using accurate table mode and formula-to-LaTeX. Qwen2.5-VL writes a detailed description of every figure and table. Images are saved to `data/md/assets/<paper>/`. |
| 3. Scan | `rag.scan` | Removes prompt-injection phrases before anything is indexed. |
| 4. Parse | `rag.parse_markdown` | Splits the markdown into `text`, `figure`, `table` and `equation` blocks. Each block keeps its section, number and caption. |
| 5. Chunk | `rag.make_chunks` | Each figure, table or equation becomes one chunk. Text is grouped into chunks of up to 1200 characters, one section at a time. |
| 6. Embed | `rag.embed` | Embeds the chunks with `nomic-embed-text`. Only the new paper is embedded. |
| 7. Store | `rag.save_index` | Appends the paper to `data/index.json`. The write is atomic, and a PDF that is already indexed (same SHA-256) is skipped. |

### Query features

| Feature | Where | How |
|---|---|---|
| **Ask** | `rag.ask` | Reads metadata from the question (for example "paper 2 figure 4" or "section 3.2"), filters chunks by it, runs a similarity search and asks llama3. Each answer lists its sources. |
| **Paper details** | `rag.paper_details` | A structured summary (problem, method, results and so on), plus the paper's lists of sections, figures and tables. |
| **Find papers** | `rag.find_relevant_papers` | Ranks papers by their best-matching chunk and explains why each is relevant. |
| **Compare** | `rag.compare_papers` | A comparison table written by the LLM, plus similarities and differences, for 2 or more library papers. |
| **Topic search** | `store.PaperDB.search` | Paper-level cosine search. Shows each paper's authors, extractive summary and extracted slots. |
| **Upload & compare** | `compare.compare_uploaded_paper` | Parses a new PDF without adding it to the library. Gives a TL;DR and slots, then a side-by-side comparison with the most similar library papers. |

---

## Files

```
Merged_Code/
├── app.py            # main Streamlit UI: all 6 tabs + add/remove papers in the sidebar
├── app_amal.py       # Paper Explorer UI (Topic search + Upload & compare); also standalone
├── rag.py            # ingest pipeline, index, retrieval, LLM features, CLI
├── pdf2md.py         # PDF -> rich markdown (Docling + Qwen2.5-VL descriptions)
├── parse.py          # markdown/PDF -> sections, title, authors, abstract, extractive summary
├── store.py          # PaperDB: paper-level vectors over data/index.json (FAISS or NumPy)
├── compare.py        # TL;DR (SciTLDR/BART or Ollama), slot extraction, uploaded-paper comparison
├── search.py         # CLI topic search over PaperDB
├── test_security.py  # upload-validation and prompt-injection tests
├── requirements.txt
├── models/
│   └── scitldr_model/   # optional fine-tuned BART SciTLDR checkpoint
└── data/
    ├── pdfs/         # PDFs added to the library
    ├── md/           # converted markdown + assets/<paper>/figure_XX.png, table_XX.png
    └── index.json    # the library: papers + chunks + embedding vectors
```

---

## Setup

### 1. Python packages

You need Python 3.10 or newer (this was tested with 3.13).

```bash
pip install -r requirements.txt
pip install faiss-cpu        # optional; store.py uses NumPy when it is missing
```

### 2. Ollama models

Install [Ollama](https://ollama.com), start it with `ollama serve`, and pull the three models:

```bash
ollama pull nomic-embed-text   # embeddings (required)
ollama pull llama3             # answers, summaries, comparisons (required)
ollama pull qwen2.5vl:3b       # figure/table descriptions while converting PDFs
```

If you don't pull `qwen2.5vl:3b`, PDFs still convert, but figures and tables get no description.

### 3. Optional: the SciTLDR model

Put a fine-tuned BART SciTLDR checkpoint in `models/scitldr_model/` (with `config.json`, the weights and the tokenizer files). If the folder is missing, `llama3` writes the TL;DR instead.

---

## Usage

### Web app (recommended)

```bash
streamlit run app.py
```

- **Sidebar:** upload PDFs to add them to the library, or click ✕ to remove one. Papers marked ⚠️ had injection phrases removed.
- **Ask:** for example `explain paper 2 figure 4`, `what does table 1 of paper 3 show`, `what datasets are used in section 4 of paper 1`.
- **Paper details / Find papers / Compare:** summarise one paper, find the papers that match a topic, or compare 2 or more papers.
- **Topic search / Upload & compare:** paper-level search, and comparing a PDF that isn't in the library.

To run only the Paper Explorer: `streamlit run app_amal.py`.

### Command line

```bash
# add one PDF or a whole folder (slow: Docling + VLM on every figure and table)
python rag.py add data/pdfs/
python rag.py add path/to/paper.pdf

# ask a question
python rag.py ask "explain paper 2 figure 4"

# re-index all papers from data/md after changing parsing/chunking code (no PDF conversion)
python rag.py rebuild

# convert one PDF to markdown only
python pdf2md.py paper.pdf

# paper-level topic search
python search.py "uncertainty estimation for visual ambiguity"

# compare a new PDF with the library (the PDF is NOT added)
python compare.py path/to/new_paper.pdf
```

### Tests

```bash
python -m unittest test_security -v
```

The last test sends a poisoned excerpt to the real model. It is skipped if Ollama isn't running.

---

## Configuration

These are constants at the top of each file:

| File | Setting | Default | Meaning |
|---|---|---|---|
| `rag.py` | `OLLAMA_URL` | `http://localhost:11434` (or the `OLLAMA_URL` environment variable) | Ollama server |
| `rag.py` | `EMBED_MODEL` | `nomic-embed-text` | Embedding model. If you change it, run `python rag.py rebuild`. |
| `rag.py` | `LLM_MODEL` | `llama3` | Text model, for example `llama3.2:3b` or `qwen2.5:7b` |
| `rag.py` | `CHUNK_SIZE` | `1200` | Maximum characters in a text chunk |
| `rag.py` | `NUM_CTX` | `8192` | LLM context window, in tokens |
| `rag.py` | `MAX_PDF_MB` | `50` | Upload size limit |
| `pdf2md.py` | `VLM_MODEL` | `qwen2.5vl:3b` | Vision model. Use `qwen2.5vl:7b` for better descriptions. |
| `compare.py` | `MODEL_PATH` | `models/scitldr_model` | Fine-tuned SciTLDR checkpoint |

---

## Security

Text taken from a PDF is treated as **untrusted**:

1. **Upload validation:** checks the extension, the size limit and the PDF header and EOF markers. File names are cleaned so they can't point outside the data folder.
2. **Ingest scan:** `rag.scan` removes instruction-like phrases (such as "ignore previous instructions", fake `system:` turns or `<|im_start|>` tokens) before indexing. The number removed is stored in the `flagged` field.
3. **Prompt wrapping:** every excerpt is sanitised again and placed inside `<excerpt>` tags, and the system prompt tells the model never to follow instructions found inside them.
4. Uploaded papers that you only compare go through the same validation and scan. They aren't stored: the temporary file is deleted afterwards.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama error: Connection refused` | Run `ollama serve`, then check `ollama list` shows the models above. |
| `model "llama3" not found` | Run `ollama pull llama3`, or set `LLM_MODEL` to a model you have. |
| Converting a PDF takes a long time | Expected: accurate tables, formula recognition and a VLM call for every figure and table. Use a GPU or the 3B VLM. |
| Figure descriptions say "skipped - Ollama not running" | The VLM wasn't reachable during conversion. Re-add the PDF once it is running. |
| Paper titles look wrong after a code change | Run `python rag.py rebuild`. |
| Docling fails on an uploaded PDF in **Upload & compare** | `compare.py` switches to PyMuPDF text extraction automatically. |
| `FutureWarning: The fitz API is deprecated` | Harmless warning from PyMuPDF. |
