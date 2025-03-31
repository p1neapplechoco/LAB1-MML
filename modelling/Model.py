from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from modelling.LossFunction import LossFunction, MSE

class MultipleRegression(ABC):
    def __init__(self, 
                 fit_intercept: bool = True,
                 regularization: Optional[str] = None,
                 alpha: float = 0.0) -> None:
        """
        Initialize regression model.
        
        Args:
            fit_intercept: Whether to fit intercept term
            regularization: Type of regularization ('l1', 'l2', or None)
            alpha: Regularization strength
        """
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.alpha = alpha
        self.coefficients = None
        self.intercept = 0.0
    
    @abstractmethod
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input features (polynomials, interactions, etc.) and return as pandas DataFrame"""
        pass
    
    def add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to X if required"""
        if self.fit_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X
    
    def fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve using normal equations with pseudoinverse"""
        X_transformed = self.transform_features(X)
        X_with_intercept = self.add_intercept(X_transformed)
        
        # For ridge regression (L2)
        if self.regularization == 'l2':
            # (X^T X + αI)^(-1) X^T y
            n_features = X_with_intercept.shape[1]
            identity = np.eye(n_features)
            if self.fit_intercept:
                # Don't regularize intercept
                identity[0, 0] = 0
            theta = np.linalg.inv(X_with_intercept.T @ X_with_intercept + 
                               self.alpha * identity) @ X_with_intercept.T @ y
        else:
            # Standard OLS: (X^T X)^(-1) X^T y
            theta = np.linalg.pinv(X_with_intercept) @ y
            
        return theta
    
    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray, 
                            loss_func: LossFunction,
                            learning_rate: float = 0.0001,
                            max_iter: int = 1000,
                            tol: float = 1e-6) -> np.ndarray:
        """Fit using gradient descent"""
        X_transformed = self.transform_features(X)
        X_with_intercept = self.add_intercept(X_transformed)
        
        # Initialize parameters
        theta = np.random.rand(X_with_intercept.shape[1])
        
        for i in range(max_iter):
            gradient = loss_func.gradient(X_with_intercept, y, theta)
            
            # Add regularization gradient if needed
            if self.regularization == 'l2':
                # Don't regularize intercept
                reg_gradient = np.zeros_like(theta)
                reg_gradient[1:] = self.alpha * theta[1:]
                gradient += reg_gradient
            elif self.regularization == 'l1':
                reg_gradient = np.zeros_like(theta)
                reg_gradient[1:] = self.alpha * np.sign(theta[1:])
                gradient += reg_gradient
            
            # Update parameters
            theta_new = theta - learning_rate * gradient
            
            # Check convergence
            if np.all(np.abs(theta - theta_new) < tol):
                break
                
            theta = theta_new
            
        return theta
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            log_transform: bool = True,
            method: str = 'closed_form',
            loss_func: Optional[LossFunction] = None,
            **kwargs) -> 'MultipleRegression':
        """
        Fit the regression model.
        
        Args:
            X: Feature matrix
            y: Target values
            method: 'closed_form' or 'gradient_descent'
            loss_func: Loss function (required for gradient descent)
            **kwargs: Additional parameters for gradient descent
        
        Returns:
            self: Fitted model
        """
        if log_transform:
            y_copy = np.log(y)
        else:
            y_copy = y.copy()
            
        if method == 'closed_form':
            if self.regularization == 'l1':
                raise ValueError("L1 regularization not supported with closed-form solution")
            theta = self.fit_closed_form(X, y_copy)
        elif method == 'gradient_descent':
            if loss_func is None:
                loss_func = MSE()
            theta = self.fit_gradient_descent(X, y_copy, loss_func, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract intercept and coefficients
        if self.fit_intercept:
            self.intercept = theta[0]
            self.coefficients = theta[1:]
        else:
            self.coefficients = theta
            
        return self
    
    def predict(self, X: np.ndarray, y_exp = True) -> np.ndarray:
        """Make predictions on new data"""
        if self.coefficients is None:
            raise RuntimeError("Model not fitted yet")
            
        X_transformed = self.transform_features(X)
        if y_exp:
            y_pred = np.exp(self.intercept + X_transformed @ self.coefficients)
        else:
            y_pred = self.intercept + X_transformed @ self.coefficients
        return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score"""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def score_log(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score with y in log space"""
        y_pred = np.exp(self.predict(X))
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
class StandardRegression(MultipleRegression):
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Standard regression uses features as-is
        return X.copy()

class PolynomialRegression(MultipleRegression):
    def __init__(self, degree: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.degree = degree
        
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features into polynomial features up to the specified degree"""
        result = pd.DataFrame()
        
        # Create polynomial features for each column
        for col in X.columns:
            for d in range(1, self.degree + 1):
                new_col_name = f"{col}^{d}" if d > 1 else col
                result[new_col_name] = X[col] ** d
                
        return result

class InteractionRegression(MultipleRegression):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features by adding all pairwise interaction terms"""
        # Start with a copy of the original DataFrame
        result = X.copy()
        
        # Add interaction terms
        for i, col1 in enumerate(X.columns):
            for j, col2 in enumerate(X.columns):
                # Only add interaction once (avoid duplicates)
                if j > i:
                    # Create a new column name for the interaction
                    interaction_name = f"{col1}*{col2}"
                    # Add the interaction column
                    result[interaction_name] = X[col1] * X[col2]
                    
        return result

class MixedRegression(MultipleRegression):
    def __init__(self, 
                 degree: int = 2,
                 include_interactions: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.degree = degree
        self.include_interactions = include_interactions
        
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features into polynomial features and interaction terms"""
        # Start with a copy of the original DataFrame
        result = X.copy()
        
        # Add polynomial terms
        for col in X.columns:
            for d in range(2, self.degree + 1):
                new_col_name = f"{col}^{d}"
                result[new_col_name] = X[col] ** d
                
        # Add interaction terms if requested
        if self.include_interactions:
            for i, col1 in enumerate(X.columns):
                for j, col2 in enumerate(X.columns):
                    # Only add interaction once (avoid duplicates)
                    if j > i:
                        # Create a new column name for the interaction
                        interaction_name = f"{col1}*{col2}"
                        # Add the interaction column
                        result[interaction_name] = X[col1] * X[col2]
                        
        return result