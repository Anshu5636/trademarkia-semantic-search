import numpy as np
from scipy.spatial.distance import cdist


class FuzzyCMeans:
    def __init__(self, n_clusters=20, m=2.0, max_iter=150, tol=1e-4, random_state=42):
        self.n_clusters   = n_clusters
        self.m            = m
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        rng = np.random.default_rng(self.random_state)
        n   = X.shape[0]
        k   = self.n_clusters
        U   = rng.random((n, k)).astype(np.float32)
        U   = U / U.sum(axis=1, keepdims=True)

        for iteration in range(self.max_iter):
            Um      = U ** self.m
            centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
            dist    = cdist(X, centers, metric="euclidean").astype(np.float32)
            dist    = np.fmax(dist, 1e-10)
            exp     = 2.0 / (self.m - 1.0)
            tmp     = dist[:, :, None] / dist[:, None, :]
            U_new   = 1.0 / (tmp ** exp).sum(axis=2)
            U_new   = U_new / U_new.sum(axis=1, keepdims=True)
            delta   = np.linalg.norm(
                centers - (U ** self.m).T @ X / (U ** self.m).sum(axis=0)[:, None]
            )
            U = U_new
            if delta < self.tol:
                break

        self.centers_ = centers
        self.U_       = U
        self.fpc_     = float((U ** 2).sum() / n)
        return self

    def predict_soft(self, X_new: np.ndarray) -> np.ndarray:
        dist = cdist(X_new, self.centers_, metric="euclidean").astype(np.float32)
        dist = np.fmax(dist, 1e-10)
        exp  = 2.0 / (self.m - 1.0)
        tmp  = dist[:, :, None] / dist[:, None, :]
        U    = 1.0 / (tmp ** exp).sum(axis=2)
        return U / U.sum(axis=1, keepdims=True)