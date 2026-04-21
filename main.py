import sys
import re
import numpy as np
import pandas as pd

from data_loader import load_data, train_test_split
from models import ItemKNN, BPRMF
from hybrid import KGHybridRecommender, KGColdStartRecommender
from evaluation import evaluate
from knowledge_graph import MovieKnowledgeGraph
from llm import generate_sparql


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_cold_start(train, max_interactions=3, n_users=200):
    user_counts = train.groupby('userId').size()
    eligible = user_counts[user_counts >= 20].index.tolist()
    np.random.seed(42)
    cold_user_ids = np.random.choice(eligible, size=min(n_users, len(eligible)), replace=False)

    cold_rows = []
    for uid in cold_user_ids:
        user_data = train[train['userId'] == uid]
        sampled = user_data.sample(n=min(max_interactions, len(user_data)), random_state=42)
        cold_rows.append(sampled)

    non_cold = train[~train['userId'].isin(cold_user_ids)]
    cold_train = pd.concat([non_cold] + cold_rows).reset_index(drop=True)
    return cold_train, set(cold_user_ids)


def evaluate_cold_start(test, train, movies, kg, k=10, max_interactions=3, n_users=200):
    from evaluation import precision_at_k, recall_at_k, ndcg_at_k

    cold_train, cold_user_ids = simulate_cold_start(train, max_interactions, n_users)
    print(f"\nSimulated cold-start: {len(cold_user_ids)} users reduced to <={max_interactions} interactions")
    print(f"Cold training set: {len(cold_train)} ratings (vs {len(train)} original)")

    print("Retraining models on cold-start data...")
    knn_cold = ItemKNN(k=20)
    knn_cold.fit(cold_train)

    bpr_cold = BPRMF(n_factors=100, epochs=20)
    bpr_cold.fit(cold_train)

    hybrid_knn_cold = KGHybridRecommender(knn_cold, kg, cold_train, movies, alpha=0.4)
    hybrid_bpr_cold = KGHybridRecommender(bpr_cold, kg, cold_train, movies, alpha=0.4)
    kg_only_cold    = KGColdStartRecommender(kg, cold_train, movies)

    cold_models = {
        "ItemKNN": knn_cold, "BPR-MF": bpr_cold,
        "Hybrid-KNN": hybrid_knn_cold, "Hybrid-BPR": hybrid_bpr_cold,
        "KG-Only": kg_only_cold,
    }

    relevant_items = test[test['rating'] >= 4.0].groupby('userId')['movieId'].apply(list).to_dict()
    eval_users = [u for u in cold_user_ids if u in relevant_items]

    results = {}
    for name, model in cold_models.items():
        precisions, recalls, ndcgs = [], [], []
        for user_id in eval_users:
            relevant = relevant_items[user_id]
            recs = model.recommend(user_id, top_n=k)
            if not recs:
                continue
            precisions.append(precision_at_k(recs, relevant, k))
            recalls.append(recall_at_k(recs, relevant, k))
            ndcgs.append(ndcg_at_k(recs, relevant, k))
        results[name] = {
            "Precision@10": round(np.mean(precisions), 4) if precisions else 0.0,
            "Recall@10":    round(np.mean(recalls), 4) if recalls else 0.0,
            "NDCG@10":      round(np.mean(ndcgs), 4) if ndcgs else 0.0,
        }
    return results


def print_results_table(title, results, model_names):
    print(f"\n{title}")
    header = f"{'Metric':<15}" + "".join(f"{n:>14}" for n in model_names)
    print(header)
    print("-" * len(header))
    for metric in ["Precision@10", "Recall@10", "NDCG@10"]:
        row = f"{metric:<15}" + "".join(f"{results[n][metric]:>14}" for n in model_names)
        print(row)


def _run_fallback(kg, user_input):
    all_genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                  "Horror", "IMAX", "Musical", "Mystery", "Romance",
                  "Sci-Fi", "Thriller", "War", "Western"]
    matched = [g for g in all_genres
               if g.lower() in user_input.lower()
               or g.replace("-", " ").lower() in user_input.lower()]
    if not matched:
        return []
    if len(matched) > 1:
        return kg.query_movies_by_multiple_genres(matched, limit=200)
    return kg.query_by_genre(matched[0], limit=200)


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive demo
# ═══════════════════════════════════════════════════════════════════════════════

PREFIXES = """
    PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX mr:     <http://movierec.org/>
    PREFIX schema: <http://schema.org/>
"""


def query_kg_via_llm(kg, user_input):
    """Run the full LLM -> SPARQL -> KG pipeline. Returns list of (movieId, title)."""
    query, error = generate_sparql(user_input)

    if error:
        print(f"  [LLM Error] {error}")
        print(f"  [Fallback] Using direct KG genre lookup...")
        fallback = _run_fallback(kg, user_input)
        return fallback, fallback  

    if "PREFIX" not in query:
        query = PREFIXES + query

    print(f"\n  Generated SPARQL:")
    for line in query.strip().split("\n"):
        print(f"    {line.strip()}")

    try:
        results = kg.g.query(query)
        rows = [(int(r.movieId), str(r.title)) for r in results]
        seen = set()
        unique = []
        for mid, title in rows:
            if mid not in seen:
                seen.add(mid)
                unique.append((mid, title))

        # Also run a broader query (LIMIT 200) for the hybrid injection pool
        broad_query = re.sub(r'LIMIT\s+\d+', 'LIMIT 200', query)
        broad_results = kg.g.query(broad_query)
        broad_rows = []
        broad_seen = set()
        for r in broad_results:
            mid = int(r.movieId)
            if mid not in broad_seen:
                broad_seen.add(mid)
                broad_rows.append((mid, str(r.title)))

        return unique, broad_rows
    except Exception as e:
        print(f"  [Query failed] {e}")
        print(f"  [Fallback] Using direct KG genre lookup...")
        fallback = _run_fallback(kg, user_input)
        return fallback, fallback


def interactive_demo(kg, knn, hybrid_knn, train, movies):
    """Interactive recommendation demo showing the full pipeline."""

    # Build movie title lookup
    title_lookup = dict(zip(movies['movieId'], movies['title']))

    # Pick a demo user with a decent number of ratings
    user_counts = train.groupby('userId').size().sort_values(ascending=False)
    demo_users = user_counts.head(20).index.tolist()

    # Get genre profile for display
    movie_genres = {}
    for _, row in movies.iterrows():
        genres = row['genres']
        if genres and genres != "(no genres listed)":
            movie_genres[row['movieId']] = genres.split("|")
        else:
            movie_genres[row['movieId']] = []

    print("\n" + "=" * 70)
    print("  INTERACTIVE MOVIE RECOMMENDATION DEMO")
    print("=" * 70)
    print("\nThis demo shows the full pipeline:")
    print("  1. Your natural language query")
    print("  2. LLM generates a SPARQL query")
    print("  3. Knowledge Graph returns genre-matched movies")
    print("  4. Hybrid model re-ranks using CF + KG signals")
    print(f"\nType a movie request (or 'quit' to exit)")
    print(f"Examples:")
    print(f"  'recommend me some thriller movies'")
    print(f"  'I want sci-fi and action movies'")
    print(f"  'show me horror and mystery films'")
    print(f"  'comedy movies please'")
    print("-" * 70)

    # Select a consistent demo user
    demo_user = demo_users[0]
    user_ratings = train[train['userId'] == demo_user].sort_values('rating', ascending=False)
    top_rated = user_ratings.head(5)

    print(f"\nDemo User #{demo_user} — their top-rated movies:")
    for _, row in top_rated.iterrows():
        mid = row['movieId']
        genres = ", ".join(movie_genres.get(mid, []))
        print(f"  {title_lookup.get(mid, 'Unknown'):50s}  ({genres})")

    # Get user's genre profile for display
    genre_counts = {}
    for _, row in user_ratings.iterrows():
        for g in movie_genres.get(row['movieId'], []):
            genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Genre profile: {', '.join(f'{g} ({c})' for g, c in top_genres)}")
    print()

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting demo.")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Exiting demo.")
            break

        # ── Step 1: LLM -> SPARQL -> KG ──
        print(f"\n{'─' * 60}")
        print(f"  STEP 1: LLM SPARQL Generation + Knowledge Graph Query")
        print(f"{'─' * 60}")
        kg_display, kg_broad = query_kg_via_llm(kg, user_input)

        if kg_display:
            print(f"\n  KG Results ({len(kg_display)} movies shown, {len(kg_broad)} total matched):")
            for i, (mid, title) in enumerate(kg_display[:10]):
                genres = ", ".join(movie_genres.get(mid, []))
                print(f"    {i+1:2d}. {title:50s}  ({genres})")
        else:
            print("  No results from KG.")

        # ── Step 2: CF-only recommendations ──
        print(f"\n{'─' * 60}")
        print(f"  STEP 2: ItemKNN (CF-only) Recommendations for User #{demo_user}")
        print(f"{'─' * 60}")
        cf_recs = knn.recommend(demo_user, top_n=10)
        for i, mid in enumerate(cf_recs):
            title = title_lookup.get(mid, f"Movie {mid}")
            genres = ", ".join(movie_genres.get(mid, []))
            print(f"    {i+1:2d}. {title:50s}  ({genres})")

        # ── Step 3: Hybrid (CF + KG) recommendations ──
        # Use the broad KG pool (200 movies) for injection so there are
        # enough unrated candidates even for heavy users
        print(f"\n{'─' * 60}")
        print(f"  STEP 3: Hybrid-KNN (CF + KG) Recommendations for User #{demo_user}")
        print(f"{'─' * 60}")
        kg_broad_ids = {m for m, _ in kg_broad} if kg_broad else set()
        kg_broad_list = [m for m, _ in kg_broad] if kg_broad else None
        hybrid_recs = hybrid_knn.recommend(demo_user, top_n=10, kg_movie_ids=kg_broad_list)
        for i, mid in enumerate(hybrid_recs):
            title = title_lookup.get(mid, f"Movie {mid}")
            genres = ", ".join(movie_genres.get(mid, []))
            marker = " <-- KG match" if mid in kg_broad_ids else ""
            print(f"    {i+1:2d}. {title:50s}  ({genres}){marker}")

        # ── Summary ──
        if kg_broad:
            cf_overlap = len(set(cf_recs) & kg_broad_ids)
            hybrid_overlap = len(set(hybrid_recs) & kg_broad_ids)
            print(f"\n  Summary: Hybrid surfaced {hybrid_overlap} genre-matched movies "
                  f"vs {cf_overlap} from pure CF.")

        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = "full"  # default: run eval + demo
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    # ── Data Loading ──
    ratings, movies = load_data(sample_users=5000)
    train, test = train_test_split(ratings)

    # ── Build Knowledge Graph ──
    kg = MovieKnowledgeGraph()
    kg.build(movies)

    # ── Train Base Models ──
    knn = ItemKNN(k=20)
    knn.fit(train)

    bpr = BPRMF(n_factors=100, epochs=40)
    bpr.fit(train)

    # ── Build Hybrid Models ──
    print("\n--- Building Hybrid Models ---")
    hybrid_knn = KGHybridRecommender(knn, kg, train, movies, alpha=0.7)
    hybrid_bpr = KGHybridRecommender(bpr, kg, train, movies, alpha=0.7)
    kg_only    = KGColdStartRecommender(kg, train, movies)

    all_models = {
        "ItemKNN": knn, "BPR-MF": bpr,
        "Hybrid-KNN": hybrid_knn, "Hybrid-BPR": hybrid_bpr,
        "KG-Only": kg_only,
    }
    model_names = list(all_models.keys())

    if mode in ("full", "eval"):
        # ── Full Evaluation ──
        print("\n--- Evaluating All Models (200 users) ---")
        full_results = {}
        for name, model in all_models.items():
            full_results[name] = evaluate(model, test, train)
        print_results_table("Full Evaluation Results:", full_results, model_names)

        # ── Cold-Start Evaluation ──
        print("\n--- Cold-Start Evaluation (simulated: 200 users reduced to <=3 interactions) ---")
        cold_results = evaluate_cold_start(test, train, movies, kg, max_interactions=3, n_users=200)
        print_results_table("Cold-Start Results:", cold_results, model_names)

    if mode in ("full", "demo"):
        # ── Interactive Demo ──
        interactive_demo(kg, knn, hybrid_knn, train, movies)
    elif mode == "eval":
        print("\nRun 'python3 main.py demo' to skip eval and go straight to interactive mode.")