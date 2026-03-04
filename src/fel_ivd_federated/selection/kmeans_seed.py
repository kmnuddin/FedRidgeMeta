import numpy as np
from sklearn.cluster import KMeans

def kmeans_seed(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    if k <= 0 or X.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    k = min(k, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labs = km.fit_predict(X)
    centers = km.cluster_centers_
    idxs = []
    for j in range(k):
        mask = np.where(labs == j)[0]
        if mask.size == 0:
            continue
        C = centers[j][None, :]
        i = mask[np.argmin(((X[mask] - C)**2).sum(axis=1))]
        idxs.append(int(i))
    return np.array(idxs, dtype=int)
