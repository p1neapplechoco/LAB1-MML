import numpy as np
from .LossFunction import LossFunction

class Model:
    def __init__(self, regularization=None):
        self.weights = None
        self.regularization = regularization
    
    def fit(self, X, y):
        # Abstract method to be implemented by subclasses
        raise NotImplementedError
    
    def predict(self, X):
        # Common prediction logic
        return X @ self.weights
    
    def score(self, X, y, metric='mse'):
        # Calculate performance metrics
        # Use LossFunction class here
        y_pred = self.predict(X)
        if metric == 'mse':
            return LossFunction.mse(y, y_pred)
        elif metric == 'mae':
            return LossFunction.mae(y, y_pred)
        else:
            raise ValueError("Invalid metric. Use 'mse' or 'mae'.")