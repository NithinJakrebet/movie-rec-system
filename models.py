import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


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
        scores[list(rated_items)] = -1

        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.item_ids[i] for i in top_indices]


class BPRMF:
    def __init__(self, n_factors=20, lr=0.01, reg=0.01, epochs=10):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

    def fit(self, train):
        print("Initializing BPR-MF...")
        self.user_ids = list(train['userId'].unique())
        self.item_ids = list(train['movieId'].unique())
        self.user_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_idx = {it: i for i, it in enumerate(self.item_ids)}

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)

        self.U = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.01, (n_items, self.n_factors))

        user_items = train.groupby('userId')['movieId'].apply(set).to_dict()
        item_list = np.array(list(self.item_idx.keys()))

        print(f"Training BPR-MF for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            total_loss = 0
            samples = train.sample(frac=0.1, random_state=epoch)

            for row in samples.itertuples():
                u = self.user_idx.get(row.userId)
                i = self.item_idx.get(row.movieId)
                if u is None or i is None:
                    continue

                rated = user_items.get(row.userId, set())
                neg_candidates = [it for it in np.random.choice(item_list, 10) if it not in rated]
                if not neg_candidates:
                    continue
                j = self.item_idx[neg_candidates[0]]

                x_ui = self.U[u] @ self.V[i]
                x_uj = self.U[u] @ self.V[j]
                x_uij = x_ui - x_uj
                sigmoid = 1 / (1 + np.exp(x_uij))

                self.U[u] += self.lr * (sigmoid * (self.V[i] - self.V[j]) - self.reg * self.U[u])
                self.V[i] += self.lr * (sigmoid * self.U[u] - self.reg * self.V[i])
                self.V[j] += self.lr * (-sigmoid * self.U[u] - self.reg * self.V[j])

                total_loss += -np.log(1 / (1 + np.exp(-x_uij)) + 1e-10)

            print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.2f}")

        print("BPR-MF ready.")

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_idx:
            return []

        u = self.user_idx[user_id]
        scores = self.V @ self.U[u]
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.item_ids[i] for i in top_indices]