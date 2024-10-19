import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.beta0 = None  # Intercept
        self.beta1 = None  # Slope
        self.k=1
    
    def fit(self, X, y):
        # Flatten X if it is a 2D array with one column
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
        
        # Calculate coefficients
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        self.beta1 = numerator / denominator
        self.beta0 = y_mean - self.beta1 * X_mean
    
    def predict(self, X):
        # Flatten X if it is a 2D array with one column
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
        return self.beta0 + self.beta1 * X

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def standard_deviation(self, y_true):
        return np.std(y_true)
    
    def variance(self, y_true):
        return np.var(y_true)
    


class PolynomialRegression:
    def __init__(self, k):
        self.k = k
        self.coefficients = None
    
    def _design_matrix(self, X):
        """ Generate the design matrix for polynomial regression. """
        X_poly = np.vander(X.flatten(), self.k + 1, increasing=True)
        return X_poly
    
    def fit(self, X, y):
        """ Fit polynomial regression model using the normal equation. """
        X_poly = self._design_matrix(X)
        # Solve for coefficients using the normal equation
        self.coefficients = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    
    def predict(self, X):
        """ Predict using the fitted model. """
        X_poly = self._design_matrix(X)
        return X_poly @ self.coefficients
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def standard_deviation(self, y_true):
        return np.std(y_true)
    
    def variance(self, y_true):
        return np.var(y_true)
    
    # def save_parameters(self, filename):
    #     """ Save the model parameters to a file. """
    #     np.save(filename, self.coefficients)
    
    # def load_parameters(self, filename):
    #     """ Load the model parameters from a file. """
    #     self.coefficients = np.load(filename)

