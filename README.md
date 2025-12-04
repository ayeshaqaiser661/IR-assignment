1. Overview
This project implements a local Information Retrieval (IR) system using two classical ranking models:
TF–IDF (with cosine similarity)
Okapi BM25
The system processes a news dataset (Articles.csv), builds offline indexes, supports an interactive search interface, and evaluates retrieval performance using standard IR metrics.
This repository includes:
Document preprocessing
Index construction (TF-IDF matrix + BM25 index)
Query processing
Ranking with TF-IDF and BM25
Evaluation pipeline (Precision@5, MAP)
A terminal-based search interface
2. Setup Instructions
2.1 Create & Activate Virtual Environment
cd IR_Assignment3
python3 -m venv .venv
source .venv/bin/activate        # macOS
2.2 Install Required Libraries
pip install -r requirements.txt
Key dependencies:
pandas
scikit-learn
numpy
rank-bm25
scipy
3. Building the Indexes
Run the indexer to:
Load the dataset
Preprocess all documents
Build TF-IDF vectors
Generate BM25 index
Save everything under indexes/
python -m src.indexer
You should see output similar to:
CSV columns: ['Article', 'Date', 'Heading', 'NewsType']
Loaded 2692 documents...
Indexed documents saved in /indexes/
4. Running the Search System
Start the terminal-based search interface:
python -m src.main
Example session:
Local IR system (TF-IDF & BM25)
Type 'exit' to quit.

Query> petrol sindh government
The system prints:
Top-5 TF-IDF results
Top-5 BM25 results
Document headings
First 120 characters as a snippet
Scores for each model
Each query is also automatically stored in:
queries/labels.json
along with the Top-5 BM25 results (treated as relevant documents).
5. Evaluation Pipeline
The evaluation module uses:
Precision@5 (P@5)
Mean Average Precision (MAP)
It reads the manually curated relevance judgements from:
queries/labels.json
Run evaluation for both ranking models:
python -m src.evaluator
Sample output:
Evaluating TFIDF
Query: petrol
  P@5 = 0.600
MAP (tfidf) = 0.560

Evaluating BM25
Query: petrol
  P@5 = 1.000
MAP (bm25) = 0.833
6. Managing the Labeled Query Set
You can edit or add queries manually in:
queries/labels.json
Format:
[
  {
    "query": "petrol sindh government",
    "relevant_docs": ["doc_267", "doc_2527", "doc_544", "doc_38", "doc_1008"]
  }
]
This file is used by evaluator.py to compute P@5 and MAP.
7. Notes & Limitations
Query processing uses simple preprocessing (lowercasing + whitespace tokenization).
No stemming, lemmatization, or spelling correction.
BM25 generally performs better for short, factual queries.
TF-IDF may retrieve documents with rarer but meaningful terms.
Evaluation is based on a relatively small manually curated set of queries.