import pandas as pd
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