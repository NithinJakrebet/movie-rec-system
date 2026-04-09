from data_loader import load_data, train_test_split
from models import ItemKNN, BPRMF
from hybrid import KGHybridRecommender, KGColdStartRecommender
from evaluation import evaluate
from knowledge_graph import MovieKnowledgeGraph
from llm import generate_sparql
import numpy as np


def simulate_cold_start(train, max_interactions=5, n_users=200):
    """
    Simulate cold-start by creating a reduced training set where selected
    users retain only a few interactions. Returns (cold_train, cold_users).
    """
    user_counts = train.groupby('userId').size()
    # Pick users who have enough real interactions that we can meaningfully reduce them
    eligible = user_counts[user_counts >= 20].index.tolist()
    np.random.seed(42)
    cold_user_ids = np.random.choice(eligible, size=min(n_users, len(eligible)), replace=False)

    # For cold-start users, keep only max_interactions randomly sampled ratings
    cold_rows = []
    keep_rows = []
    for uid in cold_user_ids:
        user_data = train[train['userId'] == uid]
        sampled = user_data.sample(n=min(max_interactions, len(user_data)), random_state=42)
        cold_rows.append(sampled)

    # Keep all data for non-cold-start users, reduced data for cold-start users
    non_cold = train[~train['userId'].isin(cold_user_ids)]
    import pandas as pd
    cold_train = pd.concat([non_cold] + cold_rows).reset_index(drop=True)

    return cold_train, set(cold_user_ids)


def evaluate_cold_start(models_dict, test, train, movies, kg, k=10, max_interactions=5, n_users=200):
    """
    Simulate cold-start: retrain all models on reduced data for a subset of users,
    then evaluate only those users. This shows the KG's value when CF has limited signal.
    """
    from evaluation import precision_at_k, recall_at_k, ndcg_at_k
    from models import ItemKNN, BPRMF
    from hybrid import KGHybridRecommender, KGColdStartRecommender

    cold_train, cold_user_ids = simulate_cold_start(train, max_interactions, n_users)
    print(f"\nSimulated cold-start: {len(cold_user_ids)} users reduced to <={max_interactions} interactions")
    print(f"Cold training set: {len(cold_train)} ratings (vs {len(train)} original)")

    # Retrain models on the cold-start training set
    print("Retraining models on cold-start data...")
    knn_cold = ItemKNN(k=20)
    knn_cold.fit(cold_train)

    bpr_cold = BPRMF(n_factors=100, epochs=20)
    bpr_cold.fit(cold_train)

    hybrid_knn_cold = KGHybridRecommender(knn_cold, kg, cold_train, movies, alpha=0.4)
    hybrid_bpr_cold = KGHybridRecommender(bpr_cold, kg, cold_train, movies, alpha=0.4)
    kg_only_cold    = KGColdStartRecommender(kg, cold_train, movies)

    cold_models = {
        "ItemKNN":    knn_cold,
        "BPR-MF":     bpr_cold,
        "Hybrid-KNN": hybrid_knn_cold,
        "Hybrid-BPR": hybrid_bpr_cold,
        "KG-Only":    kg_only_cold,
    }

    # Evaluate only on the cold-start users
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
            "n_evaluated":  len(precisions),
        }

    return results


def _run_fallback(kg, user_input):
    """Fall back to direct KG genre lookup when SPARQL fails."""
    print(f"  [Fallback] Using direct KG genre lookup...")
    all_genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                  "Horror", "IMAX", "Musical", "Mystery", "Romance",
                  "Sci-Fi", "Thriller", "War", "Western"]
    matched = [
        g for g in all_genres
        if g.lower() in user_input.lower()
        or g.replace("-", " ").lower() in user_input.lower()
    ]

    if not matched:
        print("    No genres detected in input")
        return

    if len(matched) > 1:
        results = kg.query_movies_by_multiple_genres(matched, limit=10)
        print(f"  Genres: {', '.join(matched)} -> {len(results)} results")
    else:
        results = kg.query_by_genre(matched[0], limit=10)
        print(f"  Genre: {matched[0]} -> {len(results)} results")

    for mid, title in results[:5]:
        print(f"    [{mid}] {title}")


if __name__ == "__main__":
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

    # ── Full Evaluation (all users) ──
    print("\n--- Evaluating All Models (200 users) ---")
    all_models = {
        "ItemKNN":        knn,
        "BPR-MF":         bpr,
        "Hybrid-KNN":     hybrid_knn,
        "Hybrid-BPR":     hybrid_bpr,
        "KG-Only":        kg_only,
    }

    full_results = {}
    for name, model in all_models.items():
        full_results[name] = evaluate(model, test, train)

    model_names = list(all_models.keys())
    header = f"{'Metric':<15}" + "".join(f"{n:>14}" for n in model_names)
    print(f"\n{header}")
    print("-" * len(header))
    for metric in ["Precision@10", "Recall@10", "NDCG@10"]:
        row = f"{metric:<15}" + "".join(f"{full_results[n][metric]:>14}" for n in model_names)
        print(row)

    # ── Cold-Start Evaluation ──
    print("\n--- Cold-Start Evaluation (simulated: 200 users reduced to <=3 interactions) ---")
    cold_results = evaluate_cold_start(all_models, test, train, movies, kg, max_interactions=3, n_users=200)

    header = f"{'Metric':<15}" + "".join(f"{n:>14}" for n in model_names)
    print(f"\n{header}")
    print("-" * len(header))
    for metric in ["Precision@10", "Recall@10", "NDCG@10"]:
        row = f"{metric:<15}" + "".join(f"{cold_results[n][metric]:>14}" for n in model_names)
        print(row)

    # ── LLM SPARQL Generation Test ──
    print("\n--- LLM SPARQL Generation ---")
    test_inputs = [
        "recommend me some thriller movies",
        "I want sci-fi and action movies",
        "show me comedy movies",
    ]

    PREFIXES = """
        PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX mr:     <http://movierec.org/>
        PREFIX schema: <http://schema.org/>
    """

    for user_input in test_inputs:
        print(f"\nInput: {user_input}")
        query, error = generate_sparql(user_input)

        if error:
            print(f"  [Error] {error}")
            _run_fallback(kg, user_input)
            continue

        if "PREFIX" not in query:
            query = PREFIXES + query

        print(f"  Generated SPARQL:\n{query}")

        try:
            results = kg.g.query(query)
            rows = [(int(r.movieId), str(r.title)) for r in results]
            # Deduplicate (LLM sometimes generates extra joins causing cross-products)
            seen = set()
            unique_rows = []
            for mid, title in rows:
                if mid not in seen:
                    seen.add(mid)
                    unique_rows.append((mid, title))
            print(f"  Results ({len(unique_rows)} movies):")
            for mid, title in unique_rows[:5]:
                print(f"    [{mid}] {title}")
        except Exception as e:
            print(f"  [Query execution failed] {e}")
            _run_fallback(kg, user_input)