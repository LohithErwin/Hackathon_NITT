import json
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
from config import STORE_DIR, EMBED_MODEL

_model = SentenceTransformer(EMBED_MODEL)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def embed(texts, is_query=False):
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype="float32")

# ---------- DB1: paper-level (papers.faiss + papers.json) ----------
class PaperDB:
    def __init__(self):
        self.idx_path = STORE_DIR / "papers.faiss"
        self.meta_path = STORE_DIR / "papers.json"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else []
        self.index = (faiss.read_index(str(self.idx_path)) if self.idx_path.exists()
                      else faiss.IndexFlatIP(_model.get_sentence_embedding_dimension()))

    def add(self, paper_id, title, authors, summary):
        if any(m["paper_id"] == paper_id for m in self.meta):
            return
        self.index.add(embed([f"{title}. {summary}"]))
        self.meta.append({"paper_id": paper_id, "title": title,
                          "authors": authors, "summary": summary})
        faiss.write_index(self.index, str(self.idx_path))
        self.meta_path.write_text(json.dumps(self.meta, indent=2))

    def search(self, query, k=3):
        n = self.index.ntotal
        if n == 0:
            return []
        D, I = self.index.search(embed([query], True), min(k, n))
        return [(self.meta[i], float(d)) for d, i in zip(D[0], I[0]) if i >= 0]

# ---------- DB2: detailed chunks (Chroma) ----------
class ChunkDB:
    def __init__(self):
        c = chromadb.PersistentClient(path=str(STORE_DIR / "chroma"))
        self.col = c.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})

    def add(self, chunks):
        if not chunks:
            return
        self.col.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embed([c["text"] for c in chunks]).tolist(),
            metadatas=[c["meta"] for c in chunks],
        )

    def search(self, query, k=5, paper_ids=None):
        where = {"paper_id": {"$in": list(paper_ids)}} if paper_ids else None
        return self.col.query(query_embeddings=embed([query], True).tolist(),
                              n_results=k, where=where,
                              include=["documents", "metadatas", "distances"])