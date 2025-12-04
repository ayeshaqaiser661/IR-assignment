import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import preprocessed_query, tokens_to_string

Base_dir = os.path.dirname(__file__)
Index_Dir = os.path.abspath(os.path.join(Base_dir, "..", "indexes"))

class IRSys:
    def __init__(self):
        with open(os.path.join(Index_Dir, "doc_ids.pkl"), "rb") as f:
            self.doc_ids = pickle.load(f)
        with open(os.path.join(Index_Dir, "tfidf_vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(Index_Dir, "tfidf_matrix.pkl"), "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(os.path.join(Index_Dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)

    def _preprocess_query(self, query):
        return preprocessed_query(query)

    def search_tfidf(self, query, top_k=10):
        tokens = self._preprocess_query(query)
        query_str = tokens_to_string(tokens)

        q_vec = self.vectorizer.transform([query_str])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        top_idx = np.argsort(-sims)[:top_k]
        results = [(self.doc_ids[i], float(sims[i])) for i in top_idx]
        return results

    def search_bm25(self, query, top_k=10):
        tokens = self._preprocess_query(query)
        scores = np.array(self.bm25.get_scores(tokens))

        top_idx = np.argsort(-scores)[:top_k]
        results = [(self.doc_ids[i], float(scores[i])) for i in top_idx]
        return results
