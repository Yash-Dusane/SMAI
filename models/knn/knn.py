import numpy as np

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.unique_labels = np.unique(y)

    def euclidean_distance(self, X1, X2):
        return np.sqrt(((X1[:, np.newaxis] - X2) ** 2).sum(axis=2))

    def manhattan_distance(self, X1, X2):
        return np.abs(X1[:, np.newaxis] - X2).sum(axis=2)

    def cosine_distance(self, X1, X2):
        dot_product = np.dot(X1, X2.T)
        norm_X1 = np.linalg.norm(X1, axis=1)
        norm_X2 = np.linalg.norm(X2, axis=1)
        return 1 - dot_product / (norm_X1[:, np.newaxis] * norm_X2)

    def predict(self, X):
        if self.distance_metric == 'euclidean':
            distances = self.euclidean_distance(X, self.X_train)
        elif self.distance_metric == 'manhattan':
            distances = self.manhattan_distance(X, self.X_train)
        elif self.distance_metric == 'cosine':
            distances = self.cosine_distance(X, self.X_train)
        else:
            raise ValueError("Unsupported distance metric")

        # Get indices of k nearest neighbors
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Predict using mode (most common label)
        predictions = np.apply_along_axis(
            lambda x: np.bincount(np.where(self.unique_labels == x[:, None])[1]).argmax(), 
            axis=1, 
            arr=k_nearest_labels
        )
        return self.unique_labels[predictions]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)