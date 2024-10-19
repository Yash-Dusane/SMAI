import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4, epsilon=1e-6, verbose=True):

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon 
        self.verbose = verbose  
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.resp_ = None  
        self.log_likelihood_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components

        log_likelihood_old = -np.inf 
        for iteration in range(self.max_iter):
            self.resp_ = self._e_step(X)

            self._m_step(X)

            self.log_likelihood_ = self._compute_log_likelihood(X)

            if self.verbose:
                print(f"Iteration {iteration + 1}: Log-Likelihood = {self.log_likelihood_:.6f}")

            if np.isnan(self.log_likelihood_):
                print(f"Warning: NaN encountered in log-likelihood at iteration {iteration + 1}.")
                break

            if np.abs(self.log_likelihood_ - log_likelihood_old) < self.tol:
                print(f"Convergence reached at iteration {iteration + 1}.")
                break

            log_likelihood_old = self.log_likelihood_

    def _e_step(self, X):
       
        log_resp = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):

            cov_k = np.diag(np.diag(self.covariances_[k])) + np.eye(X.shape[1]) * self.epsilon
            log_resp[:, k] = np.log(self.weights_[k]) + multivariate_normal.logpdf(X, mean=self.means_[k], cov=cov_k)

        max_log_resp = np.max(log_resp, axis=1, keepdims=True) 
        log_resp -= max_log_resp 
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X):
        
        N_k = np.sum(self.resp_, axis=0)
        self.weights_ = N_k / X.shape[0]
        self.means_ = np.dot(self.resp_.T, X) / N_k[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(self.resp_[:, k] * diff.T, diff) / N_k[k]

    def _compute_log_likelihood(self, X):
        
        log_likelihood = 0
        for k in range(self.n_components):
            cov_k = np.diag(np.diag(self.covariances_[k])) + np.eye(X.shape[1]) * self.epsilon
            log_likelihood += np.sum(self.weights_[k] * multivariate_normal.pdf(X, mean=self.means_[k], cov=cov_k))
        return np.log(log_likelihood)

    def getParams(self):
        
        return self.means_, self.covariances_, self.weights_

    def getMembership(self):
        
        return self.resp_

    def getLikelihood(self):
        
        return self.log_likelihood_

    def aic(self, X):
        
        n_samples, n_features = X.shape
        num_params = self._compute_num_params(n_features)
        return 2 * num_params - 2 * self.getLikelihood()

    def bic(self, X):
        
        n_samples, n_features = X.shape
        num_params = self._compute_num_params(n_features)
        return num_params * np.log(n_samples) - 2 * self.getLikelihood()

    def _compute_num_params(self, n_features):
        
        return (self.n_components * (n_features + (n_features * (n_features + 1)) / 2)) + self.n_components - 1