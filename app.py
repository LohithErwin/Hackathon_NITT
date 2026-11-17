
import requests
import streamlit as st

import app_amal
import rag
from store import PaperDB

st.set_page_config(page_title="ResearchIQ", layout="wide")
st.title("ResearchIQ")


@st.cache_data
def get_index(version):
    """Read the index from disk. `version` changes whenever the index file changes."""
    return rag.load_index()


@st.cache_resource
def get_paper_db(version):
    """Paper-level index for 'Topic search' / 'Upload & compare' (rebuilt when the library changes)."""
    return PaperDB()


version = rag.INDEX_FILE.stat().st_mtime if rag.INDEX_FILE.exists() else 0
papers, chunks, vectors = get_index(version)

# ---------------- sidebar: add papers + library ----------------
with st.sidebar:
    st.header("Add papers")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if files and st.button("Add to library", type="primary"):
        for f in files:
            with st.status(f"Processing {f.name} ...", expanded=True) as status:
                try:
                    msg = rag.add_pdf(f.name, f.getvalue(), log=st.write)
                    status.update(label=msg, state="complete")
                except ValueError as e:                       # failed validation
                    status.update(label=f"Rejected {f.name}: {e}", state="error")
                except requests.RequestException as e:
                    status.update(label=f"Ollama error: {e}", state="error")
        papers, chunks, vectors = rag.load_index()            # show the new papers right away

    st.header(f"Library ({len(papers)})")
    for pid, p in papers.items():
        c1, c2 = st.columns([5, 1])
        c1.write(f"**{p['number']}.** {p['title']}" + (" ⚠️" if p.get("flagged") else ""))
        if p.get("flagged"):
            c1.caption(f"{p['flagged']} injection phrase(s) removed")
        if c2.button("✕", key=f"del_{pid}", help="Remove this paper"):
            rag.remove_pdf(pid)
            st.rerun()

if not papers:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

names = {pid: f"{p['number']}. {p['title']}" for pid, p in papers.items()}


def run(fn, *args):
    """Call a rag function with a spinner and a friendly error if Ollama is down."""
    try:
        with st.spinner("Thinking ..."):
            return fn(*args)
    except requests.RequestException as e:
        st.error(f"Ollama error: {e}  (is 'ollama serve' running?)")


ask_tab, details_tab, find_tab, compare_tab, topic_tab, upload_tab = st.tabs(
    ["Ask", "Paper details", "Find papers", "Compare", "Topic search", "Upload & compare"])

with ask_tab:
    q = st.text_input("Question", placeholder="explain paper 2 figure 4")
    if q and (answer := run(rag.ask, q, papers, chunks, vectors)):
        st.markdown(answer)

with details_tab:
    pid = st.selectbox("Paper", list(names), format_func=names.get, key="details")
    if st.button("Summarise"):
        st.session_state["summary"] = (pid, run(rag.paper_details, pid, papers, chunks, vectors))
    # keep the result on screen when other widgets cause a rerun
    saved = st.session_state.get("summary")
    if saved and saved[0] == pid and saved[1]:
        st.markdown(saved[1])

with find_tab:
    topic = st.text_input("Topic or question", key="find")
    if topic and (found := run(rag.find_relevant_papers, topic, papers, chunks, vectors)):
        st.markdown(found)

with compare_tab:
    if len(papers) < 2:
        st.info("Add at least 2 papers to compare them.")
    else:
        picked = st.multiselect("Papers (pick 2 or more)", list(names), format_func=names.get)
        aspect = st.text_input("Compare on (optional)", key="aspect")
        if st.button("Compare"):
            if len(picked) < 2:
                st.warning("Pick at least 2 papers.")
            else:
                st.session_state["comparison"] = (
                    tuple(picked), aspect,
                    run(rag.compare_papers, picked, papers, chunks, vectors, aspect))
        saved = st.session_state.get("comparison")
        if saved and saved[0] == tuple(picked) and saved[1] == aspect and saved[2]:
            st.markdown(saved[2])

# ---------------- paper-level search + compare an unseen PDF (app_amal.py) ----------------
version = rag.INDEX_FILE.stat().st_mtime if rag.INDEX_FILE.exists() else 0
db = get_paper_db(version)

with topic_tab:
    app_amal.render_search_tab(db)

with upload_tab:
    app_amal.render_upload_tab(db)
