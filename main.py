from data_loader import load_data, train_test_split
from models import ItemKNN, BPRMF
from evaluation import evaluate
from knowledge_graph import MovieKnowledgeGraph


if __name__ == "__main__":
    # Load data
    ratings, movies = load_data(sample_users=5000)
    train, test = train_test_split(ratings)

    # Train models
    knn = ItemKNN(k=20)
    knn.fit(train)

    bpr = BPRMF(n_factors=20, epochs=10)
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

    print("\nAvailable genres:")
    print(", ".join(kg.get_genres()))

    print("\nTop Sci-Fi movies from KG:")
    for movie_id, title in kg.query_by_genre("Sci-Fi", limit=10):
        print(f"  [{movie_id}] {title}")

    print("\nTop Sci-Fi + Thriller movies from KG:")
    for movie_id, title in kg.query_movies_by_multiple_genres(["Sci-Fi", "Thriller"], limit=10):
        print(f"  [{movie_id}] {title}")