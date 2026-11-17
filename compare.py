# compare.py
import re
import sys
from pathlib import Path
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import parse
from store import PaperDB, embed

# ---------------- Model & Device Setup ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("models/scitldr_model")

# Load fine-tuned SciTLDR model if weights exist locally
sum_tokenizer = None
sum_model = None
if MODEL_PATH.exists():
    try:
        sum_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
    except Exception as e:
        print(f"[Notice] Could not load SciTLDR model from {MODEL_PATH}: {e}")


def generate_bart_tldr(text: str) -> str:
    """Generates an abstractive scientific TL;DR using the fine-tuned BART model."""
    if sum_model is None or sum_tokenizer is None:
        return ""
    inputs = sum_tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = sum_model.generate(
            inputs["input_ids"],
            max_length=64,
            min_length=15,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return sum_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def extract_slots(text: str) -> dict:
    """Extracts benchmark metrics, datasets, and model scale using regex."""
    metric_match = re.search(
        r"(?i)\b(accuracy|bleu|f1(?:\s*score)?|map|top-1|loss|auc)\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?%?)",
        text,
    )
    param_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(M|B)\s*(?:parameters|params)", text, re.IGNORECASE
    )

    datasets = [
        "ImageNet", "COCO", "GLUE", "SQuAD", "WMT", "MIMIC",
        "arXiv", "PubMed", "SlideVQA", "DUDE", "MMLongBench", "EMMA", "CogSense-Bench"
    ]
    found_datasets = [
        d for d in datasets if re.search(rf"\b{d}\b", text, re.IGNORECASE)
    ]

    return {
        "primary_metric": (
            f"{metric_match.group(1).upper()}: {metric_match.group(2)}"
            if metric_match
            else "Not explicitly reported"
        ),
        "model_scale": (
            f"{param_match.group(1)}{param_match.group(2).upper()} params"
            if param_match
            else "Not specified"
        ),
        "benchmark_datasets": (
            ", ".join(found_datasets[:2])
            if found_datasets
            else "Custom / Unspecified"
        ),
    }


def fast_pymupdf_extract(pdf_path: Path):
    """Fast in-memory text & structure extraction using PyMuPDF (avoids Docling symlink errors)."""
    with fitz.open(str(pdf_path)) as doc:
        pages_text = [page.get_text("text") for page in doc]

    full_text = "\n".join(pages_text)

    # 1. Title Heuristic: First meaningful non-empty line of Page 1
    p1_lines = [l.strip() for l in pages_text[0].split("\n") if len(l.strip()) > 5]
    title = p1_lines[0] if p1_lines else pdf_path.stem

    # 2. Abstract Extraction
    abs_match = re.search(
        r"(?i)abstract[:\s]*(.*?)(?=\n\s*(?:1\.?|I\.?|##)\s*(?:Introduction|Background))",
        full_text,
        re.DOTALL,
    )
    if abs_match:
        abstract = abs_match.group(1).replace("\n", " ").strip()
    else:
        abstract = pages_text[0][:1500].replace("\n", " ").strip()

    # 3. Key Section Extraction (Intro, Method, Results, Conclusion)
    summary_parts = [f"Abstract: {abstract}"]

    sections = [
        ("Introduction", r"(?i)(?:1\.?|I\.?)\s*Introduction[:\s]*(.*?)(?=\n\s*(?:2\.?|II\.?))"),
        ("Method", r"(?i)(?:3\.?|III\.?)\s*(?:Method|Approach|Architecture)[:\s]*(.*?)(?=\n\s*(?:4\.?|IV\.?))"),
        ("Results", r"(?i)(?:4\.?|IV\.?|5\.?|V\.?)\s*(?:Results|Experiments|Evaluation)[:\s]*(.*?)(?=\n\s*(?:5\.?|6\.?))"),
        ("Conclusion", r"(?i)Conclusion[:\s]*(.*?)(?=\n\s*(?:References|Acknowledge))"),
    ]

    for label, pattern in sections:
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            clean_snippet = " ".join(match.group(1).split())[:400]
            summary_parts.append(f"{label}: {clean_snippet}")

    extractive_summary = "\n\n".join(summary_parts)
    return title, abstract, extractive_summary


def compare_uploaded_paper(pdf_path: Path, top_k: int = 3):
    parse.validate_pdf(pdf_path)[cite: 6]

    # Step 1: Try Docling conversion; fallback to PyMuPDF on error[cite: 6]
    abstract_text = ""
    try:
        doc = parse.convert(pdf_path)[cite: 6]
        secs = parse.sections(doc)[cite: 6]
        title = parse.get_title(secs, pdf_path.stem)[cite: 6]
        summary = parse.extractive_summary(secs)[cite: 6]
        # Extract abstract for BART
        abs_found = parse._find(secs, "abstract")[cite: 6]
        abstract_text = abs_found if abs_found else summary[:1000]
    except Exception as e:
        print(f"[Notice] Docling pipeline skipped ({type(e).__name__}). Using fast in-memory PyMuPDF fallback...")
        title, abstract_text, summary = fast_pymupdf_extract(pdf_path)

    # Step 2: Generate fine-tuned SciTLDR summary
    tldr_summary = generate_bart_tldr(abstract_text)

    # Step 3: Extract parameter & metric slots
    uploaded_slots = extract_slots(summary)

    # Step 4: Query FAISS index using PaperDB[cite: 5]
    db = PaperDB()[cite: 5]
    query_repr = f"{title}. {summary}"
    query_vector = embed([query_repr], is_query=True)[cite: 5]

    num_to_fetch = min(top_k, db.index.ntotal)[cite: 5]
    distances, indices = db.index.search(query_vector, num_to_fetch)[cite: 5]

    # Step 5: Build Diff Matrix
    comparisons = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        matched_paper = db.meta[idx][cite: 5]
        matched_slots = extract_slots(matched_paper["summary"])

        comparisons.append({
            "matched_id": matched_paper["paper_id"],[cite: 5]
            "matched_title": matched_paper["title"],[cite: 5]
            "similarity_score": round(float(dist), 4),
            "summary_preview": matched_paper["summary"].split("\n\n")[0],[cite: 3]
            "full_summary": matched_paper["summary"],[cite: 3]
            "diff_matrix": {
                "Metric": {
                    "Uploaded": uploaded_slots["primary_metric"],
                    "Database": matched_slots["primary_metric"],
                },
                "Model Scale": {
                    "Uploaded": uploaded_slots["model_scale"],
                    "Database": matched_slots["model_scale"],
                },
                "Datasets": {
                    "Uploaded": uploaded_slots["benchmark_datasets"],
                    "Database": matched_slots["benchmark_datasets"],
                },
            },
        })

    response = {
        "uploaded_title": title,
        "uploaded_summary": summary,
        "uploaded_slots": uploaded_slots,
        "matches": comparisons,
    }

    if tldr_summary:
        response["tldr_summary"] = tldr_summary

    return response


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare.py <path_to_pdf>")
        sys.exit(1)

    result = compare_uploaded_paper(Path(sys.argv[1]), top_k=2)
    print("\n================== UPLOADED PAPER SUMMARY ==================")
    print(f"Title: {result['uploaded_title']}")
    if "tldr_summary" in result:
        print(f"\nSciTLDR Summary: {result['tldr_summary']}")
    print(f"\nExtracted Sections:\n{result['uploaded_summary'][:500]}...\n")

    print("================== TOP SIMILAR PAPERS IN DATABASE ==================")
    for m in result["matches"]:
        print(f"\n-> Matched: {m['matched_title']} (Similarity: {m['similarity_score']})")
        print(f"   Summary Preview: {m['summary_preview'][:200]}...")
        print("   Attribute Comparison:")
        for attr, vals in m["diff_matrix"].items():
            print(
                f"     • {attr:15}: Uploaded -> {vals['Uploaded']} | DB -> {vals['Database']}"
            )