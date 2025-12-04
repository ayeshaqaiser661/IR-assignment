import os
import json
import pandas as pd
from .search import IRSys

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
CSV_FILE = "Articles.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILE)
LABELS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "queries", "labels.json"))

df = pd.read_csv(CSV_PATH, encoding="latin1")

def describe_doc(doc_id):
    idx = int(doc_id.split("_")[1])
    row = df.iloc[idx]
    heading = str(row.get("Heading", ""))
    snippet = str(row.get("Article", ""))[:120].replace("\n", " ")
    return heading, snippet

def save_query_with_relevant(query, relevant_docs):
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)

    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    else:
        data = []

    data.append({
        "query": query,
        "relevant_docs": list(relevant_docs),
    })

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"\nQuery saved with {len(relevant_docs)} relevant docs.")


def main():
    ir = IRSys()

    print("Local IR system (TF-IDF & BM25)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        # TF-IDF results
        tfidf_results = ir.search_tfidf(query, top_k=5)
        print("\n TF-IDF results ")
        if not tfidf_results:
            print("  No results found.\n")
        else:
            for rank, (doc_id, score) in enumerate(tfidf_results, start=1):
                heading, snippet = describe_doc(doc_id)
                print(f"{rank:2d}. {doc_id}  (score={score:.4f})")
                print(f"    {heading}")
                print(f"    {snippet}...\n")

        # BM25 results
        bm25_results = ir.search_bm25(query, top_k=5)
        print("\n BM25 results")
        if not bm25_results:
            print("  No results found.\n")
        else:
            for rank, (doc_id, score) in enumerate(bm25_results, start=1):
                heading, snippet = describe_doc(doc_id)
                print(f"{rank:2d}. {doc_id}  (score={score:.4f})")
                print(f"    {heading}")
                print(f"    {snippet}...\n")

        if bm25_results:
            relevant_docs = [doc_id for doc_id, _ in bm25_results]
            save_query_with_relevant(query, relevant_docs)
        else:
            print("\nNothing to save (no BM25 results for this query).")

if __name__ == "__main__":
    main()
