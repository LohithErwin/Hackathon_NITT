# compare.py
import re
import sys
from pathlib import Path
import fitz  # PyMuPDF
import parse
from store import PaperDB, embed


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
        re.DOTALL
    )
    if abs_match:
        abstract = abs_match.group(1).replace("\n", " ").strip()
    else:
        # Fallback to first page content
        abstract = pages_text[0][:1500].replace("\n", " ").strip()

    # 3. Key Section Extraction (Intro, Method, Results, Conclusion)
    summary_parts = [f"Abstract: {abstract}"]
    
    sections = [
        ("Introduction", r"(?i)(?:1\.?|I\.?)\s*Introduction[:\s]*(.*?)(?=\n\s*(?:2\.?|II\.?))"),
        ("Method", r"(?i)(?:3\.?|III\.?)\s*(?:Method|Approach|Architecture)[:\s]*(.*?)(?=\n\s*(?:4\.?|IV\.?))"),
        ("Results", r"(?i)(?:4\.?|IV\.?|5\.?|V\.?)\s*(?:Results|Experiments|Evaluation)[:\s]*(.*?)(?=\n\s*(?:5\.?|6\.?))"),
        ("Conclusion", r"(?i)Conclusion[:\s]*(.*?)(?=\n\s*(?:References|Acknowledge))")
    ]
    
    for label, pattern in sections:
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            clean_snippet = " ".join(match.group(1).split())[:400]
            summary_parts.append(f"{label}: {clean_snippet}")

    extractive_summary = "\n\n".join(summary_parts)
    return title, extractive_summary


def compare_uploaded_paper(pdf_path: Path, top_k: int = 3):
    parse.validate_pdf(pdf_path)

    # Step 1: Try Docling conversion; if Windows symlink or HF download fails, fallback to PyMuPDF
    try:
        doc = parse.convert(pdf_path)
        secs = parse.sections(doc)
        title = parse.get_title(secs, pdf_path.stem)
        summary = parse.extractive_summary(secs)
    except Exception as e:
        print(f"[Notice] Docling pipeline skipped ({type(e).__name__}). Using fast in-memory PyMuPDF fallback...")
        title, summary = fast_pymupdf_extract(pdf_path)

    uploaded_slots = extract_slots(summary)

    # Step 2: Query FAISS index using PaperDB
    db = PaperDB()
    query_repr = f"{title}. {summary}"
    query_vector = embed([query_repr], is_query=True)

    num_to_fetch = min(top_k, db.index.ntotal)
    distances, indices = db.index.search(query_vector, num_to_fetch)

    # Step 3: Build Diff Matrix
    comparisons = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        matched_paper = db.meta[idx]
        matched_slots = extract_slots(matched_paper["summary"])

        comparisons.append({
            "matched_id": matched_paper["paper_id"],
            "matched_title": matched_paper["title"],
            "similarity_score": round(float(dist), 4),
            "summary_preview": matched_paper["summary"].split("\n\n")[0],
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

    return {
        "uploaded_title": title,
        "uploaded_summary": summary,
        "uploaded_slots": uploaded_slots,
        "matches": comparisons,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare.py <path_to_pdf>")
        sys.exit(1)

    result = compare_uploaded_paper(Path(sys.argv[1]), top_k=2)
    print("\n================== UPLOADED PAPER SUMMARY ==================")
    print(f"Title: {result['uploaded_title']}")
    print(f"\n{result['uploaded_summary'][:500]}...\n")

    print("================== TOP SIMILAR PAPERS IN DATABASE ==================")
    for m in result["matches"]:
        print(f"\n-> Matched: {m['matched_title']} (Similarity: {m['similarity_score']})")
        print(f"   Summary: {m['summary_preview'][:200]}...")
        print("   Attribute Comparison:")
        for attr, vals in m["diff_matrix"].items():
            print(
                f"     • {attr:15}: Uploaded -> {vals['Uploaded']} | DB -> {vals['Database']}"
            )