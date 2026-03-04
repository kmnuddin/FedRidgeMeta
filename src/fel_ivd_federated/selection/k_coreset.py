import numpy as np

def kcenter_greedy(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    n = X.shape[0]
    if k <= 0 or n == 0:
        return np.zeros((0,), dtype=int)
    rng = np.random.RandomState(seed)
    start = rng.randint(0, n)
    centers = [start]
    d2 = np.sum((X - X[start])**2, axis=1)
    for _ in range(1, min(k, n)):
        i = int(np.argmax(d2))
        centers.append(i)
        d2 = np.minimum(d2, np.sum((X - X[i])**2, axis=1))
    return np.array(centers, dtype=int)
