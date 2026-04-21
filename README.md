# Movie Recommender System

A hybrid movie recommendation system built for CSE 573 that combines collaborative filtering, RDF knowledge graphs, and LLM-powered SPARQL query generation. The system demonstrates how structured knowledge can enhance traditional recommendation approaches, particularly for genre-specific queries and cold-start scenarios.

---

## Project Overview

This project implements and evaluates five recommendation approaches:

- **ItemKNN** — Item-based collaborative filtering using cosine similarity
- **BPR-MF** — Bayesian Personalized Ranking with matrix factorization
- **Hybrid-KNN** — ItemKNN re-ranked with knowledge graph genre signals
- **Hybrid-BPR** — BPR-MF re-ranked with knowledge graph genre signals
- **KG-Only** — Pure knowledge graph recommendations using genre preference profiles

A local LLM (Mistral 7B) translates natural language movie requests into SPARQL queries, which are executed against an RDF knowledge graph built from MovieLens metadata. The hybrid models blend collaborative filtering scores with KG-derived genre relevance to produce personalized, query-aware recommendations.

All approaches are evaluated on Precision@10, Recall@10, and NDCG@10 under both standard and simulated cold-start conditions.

---

## Dataset

This project uses the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/).

1. Download `ml-20m.zip` from the link above
2. Unzip and place the contents in `data/ml-20m/`:

```
data/
└── ml-20m/
    ├── ratings.csv
    ├── movies.csv
    ├── tags.csv
    ├── links.csv
    └── genome-scores.csv
```

Only `ratings.csv` and `movies.csv` are required. The dataset is not included in this repo.

---

## Prerequisites

- **Python 3.9+**
- **Ollama** — for running the local LLM (Mistral 7B)

### Install Ollama

Ollama runs the local language model used for SPARQL query generation.

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**

Download from [ollama.com/download](https://ollama.com/download)

After installing, pull the Mistral model:
```bash
ollama pull mistral
```

Make sure Ollama is running before starting the demo:
```bash
ollama serve
```

This starts the Ollama API server on `http://localhost:11434`. Keep this running in a separate terminal.

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd movie-rec-system

# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate       # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

If you don't have a `requirements.txt` yet, create one with:

```
numpy
pandas
scipy
scikit-learn
rdflib
requests
```

---

## Project Structure

```
movie-rec-system/
├── main.py               # Entry point — evaluation + interactive demo
├── data_loader.py        # MovieLens data loading and train/test split
├── models.py             # ItemKNN and BPR-MF implementations
├── knowledge_graph.py    # RDF knowledge graph construction and SPARQL queries
├── hybrid.py             # KG-enhanced hybrid recommender and cold-start model
├── llm.py                # LLM SPARQL generation via Ollama (Mistral 7B)
├── evaluation.py         # Precision@10, Recall@10, NDCG@10 metrics
├── requirements.txt
├── data/
│   └── ml-20m/
│       ├── ratings.csv
│       └── movies.csv
└── README.md
```

---

## Usage

The system supports three run modes:

### Full Run (evaluation + interactive demo)
```bash
python3 main.py
```
Trains all models, prints evaluation tables for standard and cold-start settings, then drops into the interactive demo.

### Demo Only (skip evaluation)
```bash
python3 main.py demo
```
Trains models and goes straight to the interactive demo. Use this for live presentations.

### Evaluation Only (no demo)
```bash
python3 main.py eval
```
Runs full and cold-start evaluation, prints results tables, then exits.

---

## Interactive Demo

The demo showcases the full recommendation pipeline in three steps:

1. **LLM SPARQL Generation** — Type a natural language query (e.g., "give me sci-fi and action movies"). The LLM generates a SPARQL query and the knowledge graph returns genre-matched movies.
2. **CF-Only Recommendations** — ItemKNN produces its top 10 recommendations for a demo user based purely on collaborative filtering.
3. **Hybrid Recommendations** — The hybrid model combines CF candidates with KG genre-matched movies, re-ranking them using a blended score. Movies from the KG are marked with `<-- KG match`.

### Supported & Unsuppored Queries

The system recognizes the 19 MovieLens genres:

> Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western

**Example queries:**
```
recommend me some thriller movies
I want sci-fi and action movies
show me horror and mystery films
comedy movies please
give me a war drama
```

Multi-genre queries (e.g., "sci-fi and action") generate SPARQL with multiple required genre joins, returning only movies that match all specified genres.

Queries that don't mention a genre will return no KG results and fall back to pure CF recommendations.

**For Example:**
```
recommend movies longer than 1 hour
show me movies directed by Spielberg
find movies from the 1990s
something with Tom Hanks
highly rated movies
```
This is a limitation of the fallback path, not the system architecture. When Ollama is running, the LLM generates real SPARQL queries that can handle runtime, director, year, and actor constraints; but only if those attributes exist in the knowledge graph. The current KG stores movieId, title, and genres only, so even with Ollama running, non-genre constraints will return empty result.

Type `quit` or `q` to exit the demo.

---

## Evaluation

### Standard Evaluation
Evaluates all five models on 200 sampled users using 80/20 train/test split:

| Metric        | ItemKNN | BPR-MF | Hybrid-KNN | Hybrid-BPR | KG-Only |
|---------------|---------|--------|------------|------------|---------|
| Precision@10  | 0.164   | 0.063  | 0.168      | 0.064      | 0.020   |
| Recall@10     | 0.205   | 0.080  | 0.204      | 0.084      | 0.018   |
| NDCG@10       | 0.239   | 0.080  | 0.232      | 0.089      | 0.022   |

### Cold-Start Evaluation
Simulates cold-start by reducing 200 users to only 3 training interactions each, then retraining and evaluating all models on those users:

| Metric        | ItemKNN | BPR-MF | Hybrid-KNN | Hybrid-BPR | KG-Only |
|---------------|---------|--------|------------|------------|---------|
| Precision@10  | 0.071   | 0.059  | 0.058      | 0.061      | 0.020   |
| Recall@10     | 0.096   | 0.071  | 0.077      | 0.068      | 0.014   |
| NDCG@10       | 0.102   | 0.073  | 0.082      | 0.075      | 0.024   |

*Note: Results vary slightly between runs due to random initialization.*

---

## Architecture

```
User Query ("give me sci-fi movies")
        │
        ▼
┌─────────────────┐
│   LLM (Mistral) │  → Generates SPARQL query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RDF Knowledge  │  → Returns genre-matched movie IDs
│     Graph       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hybrid Model   │  → Blends CF scores + KG genre relevance
│  (CF + KG)      │     CF candidates + KG-injected candidates
└────────┬────────┘
         │
         ▼
   Ranked Top-10
   Recommendations
```

---

## Key Design Decisions

- **Schema-aware prompting**: The LLM system prompt includes the full RDF schema, available genres, and example queries to constrain SPARQL generation
- **SPARQL validation**: Generated queries are parsed with rdflib before execution; invalid queries trigger automatic retries (up to 2)
- **Fallback mechanism**: If SPARQL generation fails entirely, the system falls back to direct KG genre lookup
- **Query-aware re-ranking**: When the user specifies genres, the hybrid model injects KG results into the CF candidate pool and shifts the blend weight toward genre relevance
- **Cold-start simulation**: Rather than relying on naturally sparse users (rare in ML-20M), the system artificially reduces training data for a subset of users to evaluate cold-start robustness

---

## Troubleshooting

**"Ollama is not running"**
Make sure `ollama serve` is running in a separate terminal before starting the demo.

**LLM generates invalid SPARQL**
The system retries up to 2 times automatically. If all retries fail, it falls back to direct KG genre lookup. This is expected behavior with smaller LLMs.

**BPR-MF training is slow**
Training 40 epochs on 5000 users takes several minutes. For faster iteration during development, reduce epochs or sample fewer users in `main.py`.

**Import errors**
Make sure all `.py` files are in the same directory and your virtual environment is activated.

---

## How It Works

Imagine you're on Netflix and you ask: "show me some thriller movies." Netflix knows what you've watched before, but its recommendation engine doesn't actually understand what "thriller" means — it just knows that people who watched Movie A also watched Movie B. So when you explicitly ask for a genre, you often get recommendations that feel disconnected from your request.

This project solves that problem by combining two different approaches to recommendation:

**Approach 1 — Collaborative Filtering (CF):** This is the traditional method. It looks at your past ratings and finds movies that similar users also liked. It's great at personalization ("people like you enjoyed these") but it has no understanding of movie attributes. If you ask for horror movies and you've never watched horror, it has nothing to work with.

**Approach 2 — Knowledge Graph (KG):** We build a structured database of movie facts — every movie, its genres, and how they connect. When you say "give me horror movies," we can query this database directly and get back every horror movie in the catalog.

**The hybrid approach** merges both: the KG finds genre-relevant movies, and the CF model personalizes them based on your taste. The result is a recommendation list that respects what you asked for *and* what you're likely to enjoy.

**The LLM piece:** Instead of requiring users to write database queries by hand, we use a local language model (Mistral 7B) to translate plain English requests like "I want sci-fi and action movies" into structured SPARQL queries that the knowledge graph can execute. This makes the whole system feel conversational.

In short: you talk to the system in natural language, an AI translates your request into a database query, the knowledge graph finds matching movies, and a hybrid model personalizes the results based on your viewing history.

---

## Demo & Understanding 

 The demo takes about 3–5 minutes after the system finishes loading.

1. Open two terminal windows
2. In Terminal 1, start Ollama:
   ```bash
   ollama serve
   ```
3. In Terminal 2, navigate to the project and start the demo:
   ```bash
   cd movie-rec-system
   source myenv/bin/activate
   python3 main.py demo
   ```
4. Wait for training to complete (~3-4 minutes). You'll see the interactive prompt (`>>`) when it's ready. You are ready to demo our movie reccomender!


