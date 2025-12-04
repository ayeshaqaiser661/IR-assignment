import os
import json
from .search import IRSys

Base_Dir = os.path.dirname(__file__)
QUERIES_FILE = os.path.abspath(os.path.join(Base_Dir, "..", "queries", "labels.json"))

def precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant)
    return hits / k

def MAP(all_retrieved, all_relevant):
    ap_list = []
    for retrieved, rel_docs in zip(all_retrieved, all_relevant):
        if not rel_docs:
            continue
        hits = 0
        sum_prec = 0.0
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in rel_docs:
                hits += 1
                sum_prec += hits / i
        if hits > 0:
            ap_list.append(sum_prec / hits)
        else:
            ap_list.append(0.0)
    if not ap_list:
        return 0.0
    return sum(ap_list) / len(ap_list)

def evaluate(model="tfidf", k=5):
    ir = IRSys()

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    all_retrieved = []
    all_relevant = []
    p_at_k_list = []

    print(f"Evaluating {model.upper()} (k={k})\n")

    for item in q_data:
        query = item["query"]
        rel_docs = item["relevant_docs"]

        if model == "tfidf":
            results = ir.search_tfidf(query, top_k=k)
        else:
            results = ir.search_bm25(query, top_k=k)

        retrieved_ids = [doc_id for doc_id, _ in results]

        all_retrieved.append(retrieved_ids)
        all_relevant.append(rel_docs)

        p_k = precision_at_k(retrieved_ids, rel_docs, k)
        p_at_k_list.append(p_k)

        print(f"Query: {query}")
        print(f"Relevant:      {rel_docs}")
        print(f"Retrieved@{k}: {retrieved_ids[:k]}")
        print(f"P@{k} = {p_k:.3f}\n")

    avg_p = sum(p_at_k_list) / len(p_at_k_list) if p_at_k_list else 0.0
    map_score = MAP(all_retrieved, all_relevant)

    print(f"Average P@{k} ({model}) = {avg_p:.3f}")
    print(f"MAP ({model}) = {map_score:.3f}")
    print("-" * 50)

if __name__ == "__main__":
    evaluate("tfidf", k=5)
    print()
    evaluate("bm25", k=5)
