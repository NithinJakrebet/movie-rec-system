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
    def __init__(self, n_factors=100, lr=0.05, reg=0.01, epochs=40, lr_decay=0.97):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.lr_decay = lr_decay

    def fit(self, train):
        print("Initializing BPR-MF...")
        self.user_ids = list(train['userId'].unique())
        self.item_ids = list(train['movieId'].unique())
        self.user_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_idx = {it: i for i, it in enumerate(self.item_ids)}

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)

        # Larger init for stronger initial gradients
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))

        user_items = train.groupby('userId')['movieId'].apply(set).to_dict()

        # Pre-map to indices for speed (avoid per-row dict lookups)
        train_user_idx = train['userId'].map(self.user_idx).values
        train_item_idx = train['movieId'].map(self.item_idx).values
        train_user_ids = train['userId'].values

        current_lr = self.lr
        print(f"Training BPR-MF for {self.epochs} epochs (n_factors={self.n_factors})...")
        for epoch in range(self.epochs):
            total_loss = 0
            n_samples = 0

            # Shuffle and use 50% of data per epoch
            perm = np.random.permutation(len(train))
            sample_size = len(train) // 2
            perm = perm[:sample_size]

            for idx in perm:
                u = train_user_idx[idx]
                i = train_item_idx[idx]

                # Fast negative sampling by index
                rated = user_items.get(train_user_ids[idx], set())
                for _ in range(5):
                    j = np.random.randint(0, n_items)
                    if self.item_ids[j] not in rated:
                        break
                else:
                    continue

                x_ui = self.U[u] @ self.V[i]
                x_uj = self.U[u] @ self.V[j]
                x_uij = np.clip(x_ui - x_uj, -30, 30)
                sigmoid = 1 / (1 + np.exp(x_uij))

                self.U[u] += current_lr * (sigmoid * (self.V[i] - self.V[j]) - self.reg * self.U[u])
                self.V[i] += current_lr * (sigmoid * self.U[u] - self.reg * self.V[i])
                self.V[j] += current_lr * (-sigmoid * self.U[u] - self.reg * self.V[j])

                total_loss += -np.log(1 / (1 + np.exp(-x_uij)) + 1e-10)
                n_samples += 1

            current_lr *= self.lr_decay
            print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.2f} (lr={current_lr:.4f})")

        print("BPR-MF ready.")

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_idx:
            return []

        u = self.user_idx[user_id]
        scores = self.V @ self.U[u]
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.item_ids[i] for i in top_indices]