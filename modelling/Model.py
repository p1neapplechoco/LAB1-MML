import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from preprocessor.DataPreprocessor import DataPreprocessor  # Importing your DataPreprocessor class

class Model:
    def __init__(self, file_path, target_column='Price', train_ratio=0.4, val_ratio=0.4, random_seed=42):
        """Initialize the Model class."""
        self.file_path = file_path
        self.target_column = target_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.val_losses = []
        self.preprocessor_params = {}  # To store preprocessing parameters

    def load_data(self):
        """Load the dataset from the CSV file."""
        self.data = pd.read_csv(self.file_path)
        print("Data loaded successfully. Shape:", self.data.shape)

    def split_data(self, df):
        """Split the DataFrame into train, validation, and test sets."""
        np.random.seed(self.random_seed)
        n_samples = len(df)
        indices = np.random.permutation(n_samples)

        train_size = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        return train_df, val_df, test_df

    def preprocess_data(self):
        """Preprocess the data using DataPreprocessor."""
        # Parse non-numeric columns that should be numeric
        data = self.data.copy()
        if 'Engine' in data.columns:
            data['Engine'] = data['Engine'].str.extract(r'(\d+\.?\d*)').astype(float)
        if 'Max Power' in data.columns:
            data['Max Power'] = data['Max Power'].str.extract(r'(\d+\.?\d*)').astype(float)
        if 'Max Torque' in data.columns:
            data['Max Torque'] = data['Max Torque'].str.extract(r'(\d+\.?\d*)').astype(float)

        # Drop non-numerical columns
        non_numerical_cols = ['Make', 'Model', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Owner', 'Seller Type', 'Drivetrain']
        data = data.drop(columns=[col for col in non_numerical_cols if col in data.columns])
        print("Non-numerical columns dropped. Remaining columns:", data.columns.tolist())

        # Split the data
        train_df, val_df, test_df = self.split_data(data)

        # Initialize preprocessors
        train_preprocessor = DataPreprocessor(train_df)
        val_preprocessor = DataPreprocessor(val_df)
        test_preprocessor = DataPreprocessor(test_df)

        # Step 1: Fill missing values
        # Numeric columns: Fill with mean
        numeric_cols = [col for col in train_df.columns if is_numeric_dtype(train_df[col])]
        train_df = train_preprocessor.fill_mean(columns=numeric_cols)
        # Store means for numeric columns
        self.preprocessor_params['means'] = {col: train_df[col].mean() for col in numeric_cols}
        # Apply to val and test using train means
        for col in numeric_cols:
            val_df[col] = val_df[col].fillna(self.preprocessor_params['means'][col])
            test_df[col] = test_df[col].fillna(self.preprocessor_params['means'][col])

        # Step 2: Standardize numeric columns (skip the target column)
        numeric_cols = [col for col in train_df.columns if is_numeric_dtype(train_df[col]) and col != self.target_column]
        train_df = train_preprocessor.standard_norm(columns=numeric_cols)
        # Store means and stds for numeric columns
        self.preprocessor_params['means'] = {col: train_df[col].mean() for col in numeric_cols}
        self.preprocessor_params['stds'] = {col: train_df[col].std() for col in numeric_cols}
        # Apply to val and test using train means and stds
        for col in numeric_cols:
            std = self.preprocessor_params['stds'][col]
            if std == 0 or pd.isna(std):
                std = 1  # Prevent division by zero
            val_df[col] = (val_df[col] - self.preprocessor_params['means'][col]) / std
            test_df[col] = (test_df[col] - self.preprocessor_params['means'][col]) / std

        # Extract features and target
        self.X_train = train_df.drop(columns=[self.target_column]).values
        self.y_train = train_df[self.target_column].values
        self.X_val = val_df.drop(columns=[self.target_column]).values
        self.y_val = val_df[self.target_column].values
        self.X_test = test_df.drop(columns=[self.target_column]).values
        self.y_test = test_df[self.target_column].values

        # Fill missing values in the target
        train_mean = np.nanmean(self.y_train)  # Use nanmean to handle NaN values
        self.y_train = np.where(np.isnan(self.y_train), train_mean, self.y_train)
        self.y_val = np.where(np.isnan(self.y_val), train_mean, self.y_val)
        self.y_test = np.where(np.isnan(self.y_test), train_mean, self.y_test)

        # Ensure all data is numeric and free of NaN/inf
        self.X_train = np.array(self.X_train, dtype=np.float64)
        self.X_val = np.array(self.X_val, dtype=np.float64)
        self.X_test = np.array(self.X_test, dtype=np.float64)
        self.y_train = np.array(self.y_train, dtype=np.float64)
        self.y_val = np.array(self.y_val, dtype=np.float64)
        self.y_test = np.array(self.y_test, dtype=np.float64)

        # Check for NaN or inf
        for name, arr in [('X_train', self.X_train), ('X_val', self.X_val), ('X_test', self.X_test),
                          ('y_train', self.y_train), ('y_val', self.y_val), ('y_test', self.y_test)]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise ValueError(f"{name} contains NaN or inf values.")

        print("Data preprocessed successfully.")
        print("Train shape:", self.X_train.shape)
        print("Validation shape:", self.X_val.shape)
        print("Test shape:", self.X_test.shape)

    def train(self, learning_rate=0.001, epochs=2000):
        """Train the linear regression model using gradient descent."""
        n_samples, n_features = self.X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(epochs):
            # Forward pass
            y_pred_train = self.predict(self.X_train)
            y_pred_val = self.predict(self.X_val)

            # Compute train and validation loss (MSE)
            train_loss = np.mean((y_pred_train - self.y_train) ** 2)
            val_loss = np.mean((y_pred_val - self.y_val) ** 2)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Compute gradients
            dw = (1/n_samples) * (self.X_train.T @ (y_pred_train - self.y_train))
            db = (1/n_samples) * np.sum(y_pred_train - self.y_train)

            # Update weights
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        print("Training completed.")

    def predict(self, X):
        """Make predictions."""
        return X @ self.weights + self.bias

    def evaluate(self):
        """Evaluate the model on the test set using MSE and MAE."""
        y_pred = self.predict(self.X_test)
        mse = np.mean((y_pred - self.y_test) ** 2)
        mae = np.mean(np.abs(y_pred - self.y_test))
        print("Test MSE:", mse)
        print("Test MAE:", mae)
        return mse, mae

    def plot_loss_curve(self):
        """Plot the train-validation loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss (MSE)', color='blue')
        plt.plot(self.val_losses, label='Validation Loss (MSE)', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Train-Validation Loss Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

