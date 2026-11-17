
import numpy as np

import parse
import rag

try:
    import faiss
except ImportError:          # faiss is optional - NumPy gives the same results for small libraries
    faiss = None


def embed(texts, is_query=False):
    """Texts -> normalised float32 vectors (same model as the RAG chunks)."""
    return rag.embed(list(texts))


class _FlatIP:
    """Minimal stand-in for faiss.IndexFlatIP (inner product on normalised vectors = cosine)."""

    def __init__(self, dim):
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, queries, k):
        k = min(k, self.ntotal)
        scores = queries @ self.vectors.T
        idx = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, idx, axis=1), idx


class PaperDB:
    def __init__(self):
        papers, chunks, vectors = rag.load_index()
        self.meta, rows = [], []
        for pid, p in sorted(papers.items(), key=lambda kv: kv[1]["number"]):
            ids = [i for i, c in enumerate(chunks) if c["paper_id"] == pid and c["type"] == "text"] \
                  or [i for i, c in enumerate(chunks) if c["paper_id"] == pid]
            if not ids:
                continue
            v = vectors[ids].mean(axis=0)
            rows.append(v / np.linalg.norm(v))
            secs = parse.sections(p["text"])
            self.meta.append({"paper_id": pid, "paper_no": p["number"], "title": p["title"],
                              "authors": parse.get_authors(secs),
                              "summary": parse.extractive_summary(secs), "hash": p.get("hash")})

        dim = vectors.shape[1] if len(rows) else 768
        self.index = faiss.IndexFlatIP(dim) if faiss else _FlatIP(dim)
        if rows:
            self.index.add(np.array(rows, dtype=np.float32))

    def search(self, query, k=3):
        """[(paper_meta, cosine_score), ...] best first."""
        if self.index.ntotal == 0 or not query.strip():
            return []
        scores, idx = self.index.search(embed([query], is_query=True), min(k, self.index.ntotal))
        return [(self.meta[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
