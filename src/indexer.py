import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from rank_bm25 import BM25Okapi
from .preprocess import preprocessed_query, tokens_to_string

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
INDEX_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "indexes"))

Data = "Articles.csv"
TEXT_COLUMN = "Article"

def load_documents():
    csv_path = os.path.join(DATA_DIR, Data)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin1")

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COLUMN}' not found in CSV. Available: {list(df.columns)}"
        )

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    doc_ids = [f"doc_{i}" for i in range(len(texts))]

    print(f"Loaded {len(texts)} documents from {csv_path}")
    print("Example snippet:", texts[0][:120])

    return texts, doc_ids


def build_indexes():
    os.makedirs(INDEX_DIR, exist_ok=True)

    texts, doc_ids = load_documents()
    tokenized_docs = [preprocessed_query(t) for t in texts]
    docs_as_strings = [tokens_to_string(toks) for toks in tokenized_docs]

    vectorizer = TfidfVectorizer()
    tfidf_matrix: csr_matrix = vectorizer.fit_transform(docs_as_strings)

    with open(os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(INDEX_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)

    bm25 = BM25Okapi(tokenized_docs)
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    with open(os.path.join(INDEX_DIR, "doc_ids.pkl"), "wb") as f:
        pickle.dump(doc_ids, f)

    print(f"Indexed {len(doc_ids)} documents. Indexes saved in {INDEX_DIR}")

if __name__ == "__main__":
    build_indexes()
