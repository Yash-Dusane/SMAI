import numpy as np
import matplotlib.pyplot as plt
import pyarrow.feather as feather

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        if X.shape[0] < self.n_clusters:
            raise ValueError("Number of data points is less than the number of clusters.")

        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            self.labels_ = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)

            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                print(f" k = {self.n_clusters} => Converged after {i+1} iterations")
                break

            self.centroids = new_centroids
        else:
            print("Maximum number of iterations reached without convergence")

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        new_centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
        for i, centroid in enumerate(new_centroids):
            if np.isnan(centroid).any():
                new_centroids[i] = X[np.random.choice(X.shape[0])]
        return new_centroids

    def predict(self, X):
        return self._assign_clusters(X)

    def getCost(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances**2)