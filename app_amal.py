import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

from compare import compare_uploaded_paper, extract_slots
from store import PaperDB

# ----------------- UI Config -----------------
st.set_page_config(
    page_title="ResearchIQ - Paper Explorer",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ResearchIQ: Semantic Search & Paper Comparison")
st.markdown(
    "Query the local FAISS index or upload an unseen PDF to generate in-memory summaries "
    "and perform side-by-side baseline comparisons without persisting the document."
)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ Database Status")
    try:
        db = PaperDB()
        total_indexed = db.index.ntotal
        st.success(f"Indexed Baseline Papers: **{total_indexed}**")
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        total_indexed = 0

    st.markdown("---")
    st.markdown("### Features Active")
    st.markdown("- **FAISS Cosine Similarity** (`IndexFlatIP`)")
    st.markdown("- **BGE Embeddings** (`bge-small-en-v1.5`)")
    st.markdown("- **Fast In-Memory Extraction** (PyMuPDF)")
    st.markdown("- **Ephemeral Diff Matrix**")

# ----------------- Main Tabs -----------------
tab_search, tab_upload = st.tabs(["🔍 Semantic Topic Search", "📄 Upload & Compare Paper"])

# ================= TAB 1: Search =================
with tab_search:
    st.subheader("Search Papers by Topic or Research Query")
    query_text = st.text_input(
        "Enter research query, method name, or problem domain:",
        placeholder="e.g., visual ambiguity and uncertainty in vision language models",
    )
    top_k_search = st.slider("Number of results to retrieve:", min_value=1, max_value=5, value=3)

    if st.button("Search Database", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching FAISS index..."):
                results = db.search(query_text, k=top_k_search)

            if not results:
                st.info("No matching papers found.")
            else:
                st.success(f"Found {len(results)} matching paper(s).")
                for rank, (paper, score) in enumerate(results, start=1):
                    with st.expander(f"**Rank {rank}: {paper['title']}** (Cosine Similarity: `{score:.4f}`)", expanded=(rank == 1)):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.markdown(f"**Paper ID:** `{paper['paper_id']}`")
                            if paper.get("authors"):
                                authors_str = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else str(paper["authors"])
                                st.markdown(f"**Authors:** {authors_str[:120]}...")
                            
                            # Extract and display slots
                            slots = extract_slots(paper["summary"])
                            st.markdown("**Reported Slots:**")
                            st.markdown(f"- *Metric:* `{slots['primary_metric']}`")
                            st.markdown(f"- *Model Scale:* `{slots['model_scale']}`")
                            st.markdown(f"- *Datasets:* `{slots['benchmark_datasets']}`")
                        
                        with col_b:
                            st.markdown("**Summary:**")
                            st.info(paper["summary"])

# ================= TAB 2: Upload & Compare =================
with tab_upload:
    st.subheader("Upload an Unseen Research Paper (Ephemeral In-Memory)")
    st.caption("The PDF is processed in RAM and compared against your database baselines. It is never saved permanently.")

    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    top_k_compare = st.slider("Number of similar baseline papers to compare:", min_value=1, max_value=3, value=2)

    if uploaded_file is not None:
        if st.button("Analyze & Compare", type="primary"):
            with st.spinner("Extracting structure and running similarity search..."):
                # Save temporarily to a clean temp file for parser consumption
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_pdf_path = Path(tmp.name)

                try:
                    res = compare_uploaded_paper(temp_pdf_path, top_k=top_k_compare)
                finally:
                    # Clean up temp file immediately after analysis
                    if temp_pdf_path.exists():
                        temp_pdf_path.unlink()

            st.markdown("---")
            st.header("1. Uploaded Paper Overview")
            st.markdown(f"### **{res['uploaded_title']}**")
            
            with st.expander("Generated Abstractive / Extractive Summary", expanded=True):
                st.write(res["uploaded_summary"])

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Primary Metric", res["uploaded_slots"]["primary_metric"])
            col_s2.metric("Model Scale", res["uploaded_slots"]["model_scale"])
            col_s3.metric("Key Datasets", res["uploaded_slots"]["benchmark_datasets"])

            st.markdown("---")
            st.header("2. Similar Database Papers & Side-by-Side Comparison")

            for rank, match in enumerate(res["matches"], start=1):
                st.markdown(f"#### Match #{rank}: **{match['matched_title']}**")
                st.markdown(f"**Similarity Score:** `{match['similarity_score']}` | **Database ID:** `{match['matched_id']}`")

                # Format side-by-side comparison matrix as a pandas table
                diff_data = []
                for attr, vals in match["diff_matrix"].items():
                    diff_data.append({
                        "Attribute": attr,
                        "Uploaded Paper": vals["Uploaded"],
                        "Database Baseline": vals["Database"],
                    })

                df_diff = pd.DataFrame(diff_data)
                st.table(df_diff)

                with st.expander(f"Inspect Baseline Summary for {match['matched_id']}"):
                    st.write(match["summary_preview"])
                st.markdown("---")