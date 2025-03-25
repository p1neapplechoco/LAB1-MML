import numpy as np

class LossFunction:
    @staticmethod
    def mse(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
    
    @staticmethod
    def mae(y_true, y_pred):
        return np.abs(y_true - y_pred).mean()