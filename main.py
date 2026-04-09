from data_loader import load_data, train_test_split
from models import ItemKNN, BPRMF
from evaluation import evaluate
from knowledge_graph import MovieKnowledgeGraph
from llm import generate_sparql


if __name__ == "__main__":
    # Load data
    ratings, movies = load_data(sample_users=5000)
    train, test = train_test_split(ratings)

    # Train models
    knn = ItemKNN(k=20)
    knn.fit(train)

    bpr = BPRMF(n_factors=100, epochs=40)
    bpr.fit(train)

    # Evaluate
    print("\nEvaluating models (200 users)...")
    knn_scores = evaluate(knn, test, train)
    bpr_scores = evaluate(bpr, test, train)

    print("\n--- Results ---")
    print(f"{'Metric':<15} {'ItemKNN':>10} {'BPR-MF':>10}")
    print("-" * 37)
    for metric in ["Precision@10", "Recall@10", "NDCG@10"]:
        print(f"{metric:<15} {knn_scores[metric]:>10} {bpr_scores[metric]:>10}")

    # Knowledge graph
    kg = MovieKnowledgeGraph()
    kg.build(movies)

    # LLM SPARQL generation test
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

        # Inject prefixes if not already present
        if "PREFIX" not in query:
            query = PREFIXES + query

        print(f"  Generated SPARQL:\n{query}")

        try:
            results = kg.g.query(query)
            rows = [(int(r.movieId), str(r.title)) for r in results]
            print(f"  Results ({len(rows)} movies):")
            for mid, title in rows[:5]:
                print(f"    [{mid}] {title}")
        except Exception as e:
            print(f"  [Query execution failed] {e}")
            _run_fallback(kg, user_input)


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
        print(f"  Genres: {', '.join(matched)} → {len(results)} results")
    else:
        results = kg.query_by_genre(matched[0], limit=10)
        print(f"  Genre: {matched[0]} → {len(results)} results")

    for mid, title in results[:5]:
        print(f"    [{mid}] {title}")