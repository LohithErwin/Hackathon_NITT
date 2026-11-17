
import tempfile
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

import store
from compare import compare_uploaded_paper, extract_slots
from store import PaperDB


def render_search_tab(db):
    st.subheader("Search Papers by Topic or Research Query")
    query_text = st.text_input(
        key="amal_query", label="Enter research query, method name, or problem domain:",
        placeholder="e.g., visual ambiguity and uncertainty in vision language models",
    )
    top_k_search = st.slider("Number of results to retrieve:", min_value=1, max_value=5, value=3, key="amal_k")

    if st.button("Search Database", type="primary", key="amal_search"):
        if not query_text.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching the library..."):
                try:
                    results = db.search(query_text, k=top_k_search)
                except requests.RequestException as e:
                    st.error(f"Ollama error: {e}  (is 'ollama serve' running?)")
                    return

            if not results:
                st.info("No matching papers found.")
            else:
                st.success(f"Found {len(results)} matching paper(s).")
                for rank, (paper, score) in enumerate(results, start=1):
                    with st.expander(
                        f"**Rank {rank}: {paper['title']}** (Cosine Similarity: `{score:.4f}`)",
                        expanded=(rank == 1),
                    ):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.markdown(f"**Paper ID:** `{paper['paper_id']}`")
                            if paper.get("authors"):
                                authors_str = (
                                    ", ".join(paper["authors"])
                                    if isinstance(paper["authors"], list)
                                    else str(paper["authors"])
                                )
                                st.markdown(f"**Authors:** {authors_str[:120]}...")

                            slots = extract_slots(paper["summary"])
                            st.markdown("**Extracted Slots:**")
                            st.markdown(f"- *Metric:* `{slots['primary_metric']}`")
                            st.markdown(f"- *Model Scale:* `{slots['model_scale']}`")
                            st.markdown(f"- *Datasets:* `{slots['benchmark_datasets']}`")

                        with col_b:
                            st.markdown("**Paper Overview:**")
                            st.info(paper["summary"])


def render_upload_tab(db):
    st.subheader("Upload an Unseen Research Paper (Ephemeral In-Memory)")
    st.caption(
        "The PDF is parsed (Docling, PyMuPDF fallback), summarized (fine-tuned BART if present, "
        "otherwise Ollama) and compared against library papers. It is NOT added to the library."
    )

    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], key="amal_upload")
    top_k_compare = st.slider("Number of similar baseline papers to compare:", min_value=1, max_value=3, value=2, key="amal_top")

    if uploaded_file is not None:
        if st.button("Analyze & Compare", type="primary", key="amal_go"):
            with st.spinner("Extracting text, generating SciTLDR summary, and querying FAISS..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_pdf_path = Path(tmp.name)

                try:
                    res = compare_uploaded_paper(temp_pdf_path, top_k=top_k_compare, db=db)
                except ValueError as e:                       # failed validation / empty library
                    st.error(str(e))
                    return
                except requests.RequestException as e:
                    st.error(f"Ollama error: {e}  (is 'ollama serve' running?)")
                    return
                finally:
                    if temp_pdf_path.exists():
                        temp_pdf_path.unlink()

            st.markdown("---")
            st.header("1. Uploaded Paper Overview")
            st.markdown(f"### **{res['uploaded_title']}**")

            # Check if fine-tuned BART TLDR is available in compare.py response
            if "tldr_summary" in res:
                st.markdown(f"#### 🔬 TL;DR Summary ({res.get('tldr_source', 'SciTLDR')})")
                st.success(res["tldr_summary"])

            with st.expander("Full Extracted Abstract & Key Sections", expanded=False):
                st.write(res["uploaded_summary"])

            # Slot Metrics Bar
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Primary Metric", res["uploaded_slots"]["primary_metric"])
            col_s2.metric("Model Scale", res["uploaded_slots"]["model_scale"])
            col_s3.metric("Key Datasets", res["uploaded_slots"]["benchmark_datasets"])

            st.markdown("---")
            st.header("2. Similar Database Papers & Side-by-Side Comparison")

            for rank, match in enumerate(res["matches"], start=1):
                st.markdown(f"#### Match #{rank}: **{match['matched_title']}**")
                st.markdown(
                    f"**Cosine Similarity:** `{match['similarity_score']}` | "
                    f"**Database ID:** `{match['matched_id']}`"
                )

                # Comparison Table
                diff_data = []
                for attr, vals in match["diff_matrix"].items():
                    diff_data.append({
                        "Attribute": attr,
                        "Uploaded Paper": vals["Uploaded"],
                        "Database Baseline": vals["Database"],
                    })

                df_diff = pd.DataFrame(diff_data)
                st.table(df_diff)

                with st.expander(f"Inspect Complete Extracted Sections for {match['matched_id']}"):
                    st.write(match.get("full_summary", match["summary_preview"]))
                st.markdown("---")


def main():
    # ----------------- UI Config -----------------
    st.set_page_config(
        page_title="ResearchIQ - Paper Explorer",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 ResearchIQ: Semantic Search & Paper Comparison")
    st.markdown(
        "Query the paper library or upload an unseen PDF to generate summaries "
        "and perform side-by-side baseline comparisons without adding it to the library."
    )

    # ----------------- Sidebar -----------------
    with st.sidebar:
        st.header("⚙️ Database Status")
        try:
            db = PaperDB()
            st.success(f"Indexed Baseline Papers: **{db.index.ntotal}**")
        except Exception as e:
            st.error(f"Failed to load the paper index: {e}")
            return
        render_architecture()

    # ----------------- Main Tabs -----------------
    tab_search, tab_upload = st.tabs(["🔍 Semantic Topic Search", "📄 Upload & Compare Paper"])
    with tab_search:
        render_search_tab(db)
    with tab_upload:
        render_upload_tab(db)


def render_architecture():
    st.markdown("---")
    st.markdown("### Active Architecture")
    st.markdown("- **Cosine Similarity** (" + ("FAISS `IndexFlatIP`" if store.faiss else "NumPy flat inner product") + ")")
    st.markdown("- **Dense Embeddings** (`nomic-embed-text` via Ollama)")
    st.markdown("- **TL;DR**: fine-tuned SciTLDR (BART) if `models/scitldr_model` exists, else Ollama")
    st.markdown("- **Parser**: Docling (PyMuPDF fallback)")


if __name__ == "__main__":
    main()
