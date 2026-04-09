import numpy as np
from collections import defaultdict


class KGHybridRecommender:
    """
    Hybrid recommender that re-ranks a base CF model's candidates
    using genre-preference signals from the knowledge graph.

    For each user, it:
    1. Builds a genre preference profile from their highly-rated training items
       by querying the KG for each item's genres.
    2. At recommendation time, generates a broad candidate set from the CF model,
       scores each candidate's genre overlap with the user profile,
       and produces a blended score: alpha * CF_score + (1 - alpha) * KG_score.

    This directly addresses cold-start: users with few CF interactions still get
    meaningful signal from genre preferences, and the KG path doesn't require
    a minimum interaction threshold.
    """

    def __init__(self, base_model, kg, train, movies, alpha=0.7, candidate_pool=100):
        """
        Args:
            base_model: A fitted CF model with .recommend() and score access
            kg: A built MovieKnowledgeGraph instance
            train: Training DataFrame with userId, movieId, rating columns
            movies: Movies DataFrame with movieId, genres columns
            alpha: Blend weight (1.0 = pure CF, 0.0 = pure KG)
            candidate_pool: Number of CF candidates to re-rank
        """
        self.base_model = base_model
        self.kg = kg
        self.alpha = alpha
        self.candidate_pool = candidate_pool

        # Build movie -> genre lookup from the movies DataFrame (fast, no SPARQL needed)
        self.movie_genres = {}
        for _, row in movies.iterrows():
            mid = row['movieId']
            genres = row['genres']
            if genres and genres != "(no genres listed)":
                self.movie_genres[mid] = set(genres.split("|"))
            else:
                self.movie_genres[mid] = set()

        # Build per-user genre profiles from training data
        # Weight genres by how much the user liked the movie
        self.user_genre_profiles = {}
        high_rated = train[train['rating'] >= 3.5]
        user_groups = high_rated.groupby('userId')

        for user_id, group in user_groups:
            genre_counts = defaultdict(float)
            for _, row in group.iterrows():
                mid = row['movieId']
                rating_weight = row['rating'] / 5.0  # normalize to [0, 1]
                for genre in self.movie_genres.get(mid, []):
                    genre_counts[genre] += rating_weight

            # Normalize to a probability distribution
            total = sum(genre_counts.values())
            if total > 0:
                self.user_genre_profiles[user_id] = {
                    g: count / total for g, count in genre_counts.items()
                }
            else:
                self.user_genre_profiles[user_id] = {}

        # Track which users are in training set
        self.train_users = set(train['userId'].unique())
        self.train_items = train.groupby('userId')['movieId'].apply(set).to_dict()

        print(f"KG-Hybrid ready: alpha={alpha}, candidate_pool={candidate_pool}, "
              f"{len(self.user_genre_profiles)} user profiles built")

    def _kg_score(self, user_id, movie_id):
        """Score a movie for a user based on genre preference overlap."""
        profile = self.user_genre_profiles.get(user_id, {})
        if not profile:
            return 0.0

        movie_genres = self.movie_genres.get(movie_id, set())
        if not movie_genres:
            return 0.0

        # Sum of user's preference weights for this movie's genres
        return sum(profile.get(g, 0.0) for g in movie_genres)

    def recommend(self, user_id, top_n=10, kg_movie_ids=None):
        """Generate recommendations by blending CF and KG scores.

        Args:
            user_id: Target user
            top_n: Number of recommendations to return
            kg_movie_ids: Optional list of movie IDs from a KG query to inject
                          into the candidate pool (for query-aware recommendations)
        """
        if user_id not in self.train_users:
            return []

        cf_candidates = self.base_model.recommend(user_id, top_n=self.candidate_pool)
        if not cf_candidates:
            return []

        cf_set = set(cf_candidates)
        rated = self.train_items.get(user_id, set())
        kg_injected = []
        if kg_movie_ids:
            kg_injected = [mid for mid in kg_movie_ids
                           if mid not in cf_set and mid not in rated]

        # When KG results are injected, shift blend toward KG signal
        # so genre-matched movies can compete with CF candidates
        if kg_movie_ids:
            blend_alpha = 0.4   # query-aware: more KG weight
        else:
            blend_alpha = self.alpha  # default: more CF weight

        # Score CF candidates
        scored = []
        n_candidates = len(cf_candidates)

        for rank, movie_id in enumerate(cf_candidates):
            cf_score = 1.0 - (rank / n_candidates)
            kg_score = self._kg_score(user_id, movie_id)
            blended = blend_alpha * cf_score + (1 - blend_alpha) * kg_score
            scored.append((movie_id, blended))

        # Score KG-injected candidates
        # These are genre-relevant movies the user hasn't seen —
        # give them a CF baseline equal to a mid-ranked CF candidate
        cf_baseline = 0.5
        for movie_id in kg_injected:
            kg_score = self._kg_score(user_id, movie_id)
            blended = blend_alpha * cf_baseline + (1 - blend_alpha) * kg_score
            scored.append((movie_id, blended))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in scored[:top_n]]


class KGColdStartRecommender:
    """
    Pure KG-based recommender for cold-start evaluation.

    For users with very few ratings, uses genre preferences derived from
    whatever interactions exist, combined with item popularity as a
    tiebreaker. This demonstrates the KG's value when CF has insufficient signal.
    """

    def __init__(self, kg, train, movies):
        self.kg = kg
        self.movie_genres = {}
        for _, row in movies.iterrows():
            mid = row['movieId']
            genres = row['genres']
            if genres and genres != "(no genres listed)":
                self.movie_genres[mid] = set(genres.split("|"))
            else:
                self.movie_genres[mid] = set()

        # Item popularity (log-scaled rating count) as tiebreaker
        item_counts = train.groupby('movieId').size()
        max_count = item_counts.max()
        self.item_popularity = {
            mid: np.log1p(count) / np.log1p(max_count)
            for mid, count in item_counts.items()
        }

        # Build genre profiles
        self.user_genre_profiles = {}
        self.train_items = train.groupby('userId')['movieId'].apply(set).to_dict()

        for user_id, group in train.groupby('userId'):
            genre_counts = defaultdict(float)
            for _, row in group.iterrows():
                for genre in self.movie_genres.get(row['movieId'], []):
                    genre_counts[genre] += row['rating'] / 5.0
            total = sum(genre_counts.values())
            if total > 0:
                self.user_genre_profiles[user_id] = {
                    g: c / total for g, c in genre_counts.items()
                }

        self.all_movie_ids = list(self.movie_genres.keys())
        self.train_users = set(train['userId'].unique())

        print(f"KG-ColdStart ready: {len(self.user_genre_profiles)} user profiles")

    def recommend(self, user_id, top_n=10):
        if user_id not in self.train_users:
            return []

        profile = self.user_genre_profiles.get(user_id, {})
        if not profile:
            return []

        rated = self.train_items.get(user_id, set())

        # Score: genre overlap (primary) + popularity tiebreaker (secondary)
        scored = []
        for mid in self.all_movie_ids:
            if mid in rated:
                continue
            movie_genres = self.movie_genres.get(mid, set())
            genre_score = sum(profile.get(g, 0.0) for g in movie_genres)
            if genre_score > 0:
                pop = self.item_popularity.get(mid, 0.0)
                # 80% genre, 20% popularity
                score = 0.8 * genre_score + 0.2 * pop
                scored.append((mid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in scored[:top_n]]