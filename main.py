import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os

DATA_PATH = "data/ml-20m/"

def load_data(sample_users=5000):
    ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))

    sampled_users = ratings['userId'].drop_duplicates().sample(sample_users, random_state=42)
    ratings = ratings[ratings['userId'].isin(sampled_users)].reset_index(drop=True)

    print(f"Sampled {len(ratings)} ratings from {ratings['userId'].nunique()} users on {ratings['movieId'].nunique()} movies")
    return ratings, movies

def train_test_split(ratings, test_size=0.2):
    ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(ratings) * (1 - test_size))
    train = ratings[:split].reset_index(drop=True)
    test = ratings[split:].reset_index(drop=True)

    print(f"Train: {len(train)} ratings | Test: {len(test)} ratings")
    return train, test

# ── ItemKNN ──────────────────────────────────────────────────────
class ItemKNN:
    def __init__(self, k=20):
        self.k = k
        self.item_similarity = None
        self.user_item_matrix = None
        self.item_ids = None
        self.user_ids = None

    def fit(self, train):
        print("Building user-item matrix...")
        self.user_ids = list(train['userId'].unique())
        self.item_ids = list(train['movieId'].unique())

        user_idx = {u: i for i, u in enumerate(self.user_ids)}
        item_idx = {it: i for i, it in enumerate(self.item_ids)}

        rows = train['userId'].map(user_idx)
        cols = train['movieId'].map(item_idx)
        vals = train['rating'].values

        self.user_item_matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )

        print("Computing item-item cosine similarity...")
        self.item_similarity = cosine_similarity(self.user_item_matrix.T, dense_output=False)
        print("ItemKNN ready.")

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_ids:
            return []

        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_item_matrix[user_idx]
        rated_items = set(user_vector.nonzero()[1])

        scores = np.array(self.item_similarity.dot(user_vector.T).todense()).flatten()
        scores[list(rated_items)] = -1  # exclude already rated

        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.item_ids[i] for i in top_indices]


if __name__ == "__main__":
    ratings, movies = load_data(sample_users=5000)
    train, test = train_test_split(ratings)

    model = ItemKNN(k=20)
    model.fit(train)

    # Quick sanity check
    sample_user = train['userId'].iloc[0]
    recs = model.recommend(sample_user, top_n=10)
    rec_titles = movies[movies['movieId'].isin(recs)]['title'].tolist()

    print(f"\nTop 10 recommendations for user {sample_user}:")
    for i, title in enumerate(rec_titles, 1):
        print(f"  {i}. {title}")