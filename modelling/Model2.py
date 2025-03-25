import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union

# ========== 1. HYPOTHESIS MODEL STRATEGY ==========
class HypothesisModel(ABC):
    """Abstract base class for different hypothesis models"""
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features according to model type"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return model name"""
        pass

# Concrete Hypothesis Models
class StandardModel(HypothesisModel):
    """Standard linear model with no feature transformations"""
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
    
    def get_name(self) -> str:
        return "Standard"

class PolynomialModel(HypothesisModel):
    """Polynomial model that creates polynomial features"""
    def __init__(self, degree: int = 2):
        self.degree = degree
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        X_poly = X.copy()  # Include original features
        
        for degree in range(2, self.degree + 1):  # Start from 2 as degree 1 is original X
            for feature in range(n_features):
                new_feature = (X[:, feature].reshape(-1, 1) ** degree)
                X_poly = np.hstack((X_poly, new_feature))
        
        return X_poly
    
    def get_name(self) -> str:
        return f"Polynomial(degree={self.degree})"

class InteractionModel(HypothesisModel):
    """Model that includes interaction terms between features"""
    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        X_interact = X.copy()  # Include original features
        
        # Add interaction terms (pairwise multiplication)
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction_term = (X[:, i] * X[:, j]).reshape(-1, 1)
                X_interact = np.hstack((X_interact, interaction_term))
        
        return X_interact
    
    def get_name(self) -> str:
        return "Interaction"

class MixedModel(HypothesisModel):
    """Combined model with both polynomial and interaction terms"""
    def __init__(self, degree: int = 2):
        self.poly_model = PolynomialModel(degree=degree)
        self.interact_model = InteractionModel()
        self.degree = degree
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Get polynomial features
        X_poly = self.poly_model.transform(X)
        
        # Get interaction features (from original X)
        X_interact = self.interact_model.transform(X)
        
        # Combine and remove duplicate original features
        combined = np.hstack((X_poly, X_interact[:, X.shape[1]:]))
        
        return combined
    
    def get_name(self) -> str:
        return f"Mixed(degree={self.degree})"

# ========== 2. HYPOTHESIS MODEL FACTORY ==========
class HypothesisModelFactory:
    """Factory for creating hypothesis models"""
    @staticmethod
    def create_model(model_type: str, **kwargs) -> HypothesisModel:
        if model_type.lower() == "standard":
            return StandardModel()
        elif model_type.lower() == "polynomial":
            degree = kwargs.get("degree", 2)
            return PolynomialModel(degree=degree)
        elif model_type.lower() == "interaction":
            return InteractionModel()
        elif model_type.lower() == "mixed":
            degree = kwargs.get("degree", 2)
            return MixedModel(degree=degree)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# ========== 3. BIAS TRICK (AFFINE TRANSFORMATION) ==========
class BiasTrick:
    """Adds a bias term (column of ones) to feature matrix"""
    @staticmethod
    def apply(X: np.ndarray) -> np.ndarray:
        """Add a column of ones to the feature matrix for the bias term"""
        return np.hstack((np.ones((X.shape[0], 1)), X))

# ========== 4. LOSS FUNCTION STRATEGY ==========
class LossFunction(ABC):
    """Abstract base class for different loss functions"""
    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    weights: Optional[np.ndarray] = None) -> float:
        """Compute the loss between true and predicted values"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return loss function name"""
        pass

# Concrete Loss Functions
class MSELoss(LossFunction):
    """Mean Squared Error loss function"""
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    weights: Optional[np.ndarray] = None) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def get_name(self) -> str:
        return "MSE"

class MAELoss(LossFunction):
    """Mean Absolute Error loss function"""
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    weights: Optional[np.ndarray] = None) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    def get_name(self) -> str:
        return "MAE"

class RegularizedMSELoss(LossFunction):
    """MSE with L2 regularization"""
    def __init__(self, lambda_reg: float = 0.1):
        self.lambda_reg = lambda_reg
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    weights: Optional[np.ndarray] = None) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        if weights is not None:
            # Skip the bias term (weights[0]) in regularization
            reg_term = self.lambda_reg * np.sum(weights[1:] ** 2)
            return mse + reg_term
        return mse
    
    def get_name(self) -> str:
        return f"RegularizedMSE(lambda={self.lambda_reg})"

# ========== 5. LOSS FUNCTION FACTORY ==========
class LossFunctionFactory:
    """Factory for creating loss functions"""
    @staticmethod
    def create_loss_function(loss_type: str, **kwargs) -> LossFunction:
        if loss_type.lower() == "mse":
            return MSELoss()
        elif loss_type.lower() == "mae":
            return MAELoss()
        elif loss_type.lower() == "regularized_mse":
            lambda_reg = kwargs.get("lambda_reg", 0.1)
            return RegularizedMSELoss(lambda_reg=lambda_reg)
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

# ========== 6. REGRESSION IMPLEMENTATION STRATEGY ==========
class RegressionImplementation(ABC):
    """Abstract base class for regression implementation"""
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the model and return weights"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return implementation name"""
        pass

# Concrete Regression Implementation
class PseudoInverseRegression(RegressionImplementation):
    """Regression using pseudo-inverse matrix solution"""
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit using analytical solution: (X^T @ X)^-1 @ X^T @ y"""
        X_T = X.T
        XTX = X_T @ X
        XTX_inv = np.linalg.inv(XTX)
        pseudo_inverse = XTX_inv @ X_T
        weights = pseudo_inverse @ y
        return weights
    
    def get_name(self) -> str:
        return "PseudoInverse"

# ========== 7. REGRESSION IMPLEMENTATION FACTORY ==========
class RegressionImplementationFactory:
    """Factory for creating regression implementations"""
    @staticmethod
    def create_implementation(implementation_type: str) -> RegressionImplementation:
        if implementation_type.lower() == "pseudo_inverse":
            return PseudoInverseRegression()
        else:
            raise ValueError(f"Unknown regression implementation: {implementation_type}")

# ========== 8. TRAINING COMPONENT ==========
class Trainer:
    """Handles model training and evaluation"""
    def __init__(self, 
                hypothesis_model: HypothesisModel,
                loss_function: LossFunction,
                regression_implementation: RegressionImplementation):
        self.hypothesis_model = hypothesis_model
        self.loss_function = loss_function
        self.regression_implementation = regression_implementation
        self.weights = None
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the model using the analytical solution"""
        # 1. Transform features according to the hypothesis model
        X_transformed = self.hypothesis_model.transform(X_train)
        
        # 2. Apply bias trick (affine transformation)
        X_with_bias = BiasTrick.apply(X_transformed)
        
        # 3. Fit using regression implementation
        self.weights = self.regression_implementation.fit(X_with_bias, y_train)
        
        # 4. Calculate training loss
        y_pred_train = self.predict(X_train)
        train_loss = self.loss_function.compute_loss(y_train, y_pred_train, self.weights)
        self.history['train_loss'].append(train_loss)
        
        # 5. Calculate validation loss if provided
        if X_val is not None and y_val is not None:
            y_pred_val = self.predict(X_val)
            val_loss = self.loss_function.compute_loss(y_val, y_pred_val, self.weights)
            self.history['val_loss'].append(val_loss)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.weights is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Same transformation pipeline as during training
        X_transformed = self.hypothesis_model.transform(X)
        X_with_bias = BiasTrick.apply(X_transformed)
        
        # Make predictions
        return X_with_bias @ self.weights
    
    def plot_train_val_loss(self) -> None:
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.history['train_loss'], label='Training Loss', marker='o')
        if len(self.history['val_loss']) > 0:
            plt.plot(self.history['val_loss'], label='Validation Loss', marker='s')
        
        plt.title('Train/Val Loss Plot')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def display_model_weights(self) -> None:
        """Display model weights"""
        if self.weights is None:
            print("Model not trained yet.")
            return
        
        print("Model Weights:")
        print(f"Bias: {self.weights[0]}")
        for i, w in enumerate(self.weights[1:], 1):
            print(f"Weight {i}: {w}")

# ========== 9. MAIN MODELING SYSTEM ==========
class ModelingSystem:
    """Main system that integrates all components"""
    def __init__(self):
        self.model_factory = HypothesisModelFactory()
        self.loss_factory = LossFunctionFactory()
        self.regression_factory = RegressionImplementationFactory()
        self.trainer = None
    
    def create_model(self, model_type: str, loss_type: str, 
                    regression_type: str = "pseudo_inverse", **kwargs) -> None:
        """Create a complete model using factories"""
        hypothesis_model = self.model_factory.create_model(model_type, **kwargs)
        loss_function = self.loss_factory.create_loss_function(loss_type, **kwargs)
        regression_implementation = self.regression_factory.create_implementation(regression_type)
        
        self.trainer = Trainer(hypothesis_model, loss_function, regression_implementation)
        
        print(f"Created model: {hypothesis_model.get_name()} with {loss_function.get_name()} "
              f"using {regression_implementation.get_name()} implementation")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the model"""
        if self.trainer is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        self.trainer.train(X_train, y_train, X_val, y_val)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.trainer is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        return self.trainer.predict(X)
    
    def display_model_weights(self) -> None:
        """Display model weights"""
        if self.trainer is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        self.trainer.display_model_weights()
    
    def plot_train_val_loss(self) -> None:
        """Plot train/val loss"""
        if self.trainer is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        self.trainer.plot_train_val_loss()

# ========== EXAMPLE USAGE ==========
def example_usage():
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = 3 + 2 * X[:, 0] + 1.5 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1] + np.random.randn(100) * 0.1
    
    # Split data into train/validation sets
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    # Create and use modeling system
    modeling_system = ModelingSystem()
    
    # Create model
    modeling_system.create_model(
        model_type="interaction",
        loss_type="regularized_mse",
        regression_type="pseudo_inverse",
        lambda_reg=0.1
    )
    
    # Train model
    modeling_system.train(X_train, y_train, X_val, y_val)
    
    # Display results
    modeling_system.display_model_weights()
    
    # Make predictions
    y_pred = modeling_system.predict(X_val)
    
    # Plot loss
    modeling_system.plot_train_val_loss()

if __name__ == "__main__":
    example_usage()