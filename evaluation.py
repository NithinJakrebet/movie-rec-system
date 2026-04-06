import numpy as np


def precision_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]
    hits = len(set(recommended) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0
    recommended = recommended[:k]
    hits = len(set(recommended) & set(relevant))
    return hits / len(relevant)

def ndcg_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]
    relevant_set = set(relevant)
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended)
        if item in relevant_set
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(model, test, train, k=10, n_users=200):
    relevant_items = test[test['rating'] >= 4.0].groupby('userId')['movieId'].apply(list).to_dict()
    train_users = set(train['userId'].unique())

    precisions, recalls, ndcgs = [], [], []
    eval_users = [u for u in list(relevant_items.keys()) if u in train_users][:n_users]

    for user_id in eval_users:
        relevant = relevant_items[user_id]
        recs = model.recommend(user_id, top_n=k)
        if not recs:
            continue
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return {
        "Precision@10": round(np.mean(precisions), 4),
        "Recall@10":    round(np.mean(recalls), 4),
        "NDCG@10":      round(np.mean(ndcgs), 4),
    }