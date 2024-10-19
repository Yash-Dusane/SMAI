import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from mpl_toolkits.mplot3d import Axes3D

class PCAClass:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def explained_variance(self, X):
        X_centered = X - self.mean
        total_variance = np.var(X_centered, axis=0).sum()
        projected_data = self.transform(X)
        variance_explained = np.var(projected_data, axis=0)
        return variance_explained / total_variance
    
    def checkPCA(self, X):
        X_transformed = self.transform(X)
        return X_transformed.shape[1] == self.n_components