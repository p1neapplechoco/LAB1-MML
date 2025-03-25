from Model import Model
import numpy as np

class LinearRegression(Model):
    def fit(self, X, y):
        # Implement analytical solution (X^T @ X)^-1 @ X^T @ y
        X_t = X.T
        self.weights = np.linalg.inv(X_t @ X) @ X_t @ y
        return self