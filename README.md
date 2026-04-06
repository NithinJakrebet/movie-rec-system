# Movie Recommender System

A terminal-based movie recommendation system built for CSE 573. The system combines classical collaborative filtering baselines with a knowledge graph and LLM-powered query generation to produce and evaluate ranked movie recommendations.

---

## Project Goals

- Implement classical recommender baselines (ItemKNN, BPR-MF) and evaluate them on standard ranking metrics
- Build an RDF knowledge graph from MovieLens and movie metadata
- Use a local LLM (Mistral 7B or Llama 3.1 8B) to generate SPARQL queries from natural language input
- Evaluate all approaches on Precision@10, Recall@10, and NDCG@10
- Demonstrate cold-start robustness of the knowledge graph approach vs. collaborative filtering

---

## Dataset

This project uses the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/).

Download it and place the contents in `data/ml-20m/`:

data/
└── ml-20m/
├── ratings.csv
├── movies.csv
└── ...

The dataset is not included in this repo. Do not commit it.

---

## Setup

Requires Python 3.9+.
```bash
python -m venv venv

source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Running the Project
```bash
python main.py
```

This will load the dataset, train the model, and output recommendations to the terminal.