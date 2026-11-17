# search.py
import sys
from store import PaperDB

def search_papers(query: str, top_k: int = 3):
    db = PaperDB()
    results = db.search(query, k=top_k)
    return results

def print_results(query: str, results: list):
    print(f"\nQuery: '{query}'")
    print("=" * 60)
    if not results:
        print("No matching papers found.")
        return

    for rank, (paper, score) in enumerate(results, start=1):
        print(f"\n[Rank {rank}] Cosine Similarity: {score:.4f}")
        print(f"Paper ID : {paper['paper_id']}")
        print(f"Title    : {paper['title']}")
        if paper.get("authors"):
            print(f"Authors  : {', '.join(paper['authors'])[:90]}...")
        # Print abstract preview
        first_section = paper['summary'].split("\n\n")[0]
        print(f"Summary  : {first_section[:300]}...")

if __name__ == "__main__":
    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "multimodal document QA with interleaved visual elements"
    matches = search_papers(query_text, top_k=3)
    print_results(query_text, matches)