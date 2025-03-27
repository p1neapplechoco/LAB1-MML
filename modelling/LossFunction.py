from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Union, List


class LossFunction(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the loss between true values and predictions"""
        pass
    
    @abstractmethod
    def gradient(self, X: np.ndarray, y_true: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute gradient for parameter updates"""
        pass

class MSE(LossFunction):
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, X: np.ndarray, y_true: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # Gradient of MSE: -2/m * X^T * (y - X*theta)
        m = X.shape[0]
        return -2/m * X.T @ (y_true - X @ theta)

class MAE(LossFunction):
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    def gradient(self, X: np.ndarray, y_true: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # Gradient of MAE: -1/m * X^T * sign(y - X*theta)
        m = X.shape[0]
        return -1/m * X.T @ np.sign(y_true - X @ theta)