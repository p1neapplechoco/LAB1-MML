import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import os

class CustomLogTransformer:
    """Custom transformer for log transformation"""
    def __init__(self, base=np.e):
        self.base = base
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return np.log1p(X) / np.log(self.base)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def inverse_transform(self, X):
        return np.power(self.base, X) - 1

def preprocess_data(data_path=None, data=None, save_path=None, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Preprocesses the data using scikit-learn pipelines to avoid data leakage.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file. If provided, data will be loaded from this path.
    data : pandas.DataFrame, optional
        Data to preprocess. If provided, data_path will be ignored.
    save_path : str, optional
        Path to save the preprocessed data. If not provided, data will not be saved.
    
    Returns:
    --------
    train_df, val_df, test_df : pandas.DataFrame
        Preprocessed train, validation, and test datasets.
    preprocessor : ColumnTransformer
        Fitted preprocessor for future transformations.
    """
    if (train_ratio + val_ratio + test_ratio != 1):
        raise ValueError("Ratio must sum up to 1")

    # Load data if path is provided
    if data is None and data_path is not None:
        data = pd.read_csv(data_path)
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Split data
    train_df, temp_df = train_test_split(data, train_size=train_ratio, random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=(val_ratio) / (val_ratio + test_ratio), random_state=42)
    
    # Define column types
    log_columns = ['Price']
    minmax_columns = ['Year', 'Kilometer']
    ordinal_columns = ['Owner']
    onehot_columns = ['Drivetrain', 'Fuel Type', 'Seller Type', 'Transmission']
    standard_columns = ['Max Power RPM', 'Max Torque RPM']
    
    # Create owner categories for ordinal encoding
    owner_categories = [["UnRegistered Car", "First", "Second", "Third", "Fourth", "4 or More"]]
    
    # Define preprocessing pipelines
    log_transformer = Pipeline([
        ('log', CustomLogTransformer(base=np.e))
    ])
    
    minmax_transformer = Pipeline([
        ('minmax', MinMaxScaler())
    ])
    
    standard_transformer = Pipeline([
        ('standard', StandardScaler())
    ])
    
    ordinal_transformer = Pipeline([
        ('ordinal', OrdinalEncoder(categories=owner_categories))
    ])
    
    onehot_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('log', log_transformer, log_columns),
            ('minmax', minmax_transformer, minmax_columns),
            ('standard', standard_transformer, standard_columns),
            ('ordinal', ordinal_transformer, ordinal_columns),
            ('onehot', onehot_transformer, onehot_columns)
        ],
        remainder='passthrough'
    )
    
    # Get column names for the transformed data
    def get_feature_names(column_transformer):
        # Get output feature names for all transformers
        output_features = []
        
        # Process log columns (stay the same, just log transformed)
        output_features.extend(log_columns)
        
        # Process minmax columns (stay the same, just scaled)
        output_features.extend(minmax_columns)
        
        # Process standard columns (stay the same, just standardized)
        output_features.extend(standard_columns)
        
        # Process ordinal columns (stay the same, just encoded as numbers)
        output_features.extend(ordinal_columns)
        
        # Get all categories from one-hot encoding
        for col in onehot_columns:
            # Get unique values in the training data
            unique_values = sorted(train_df[col].unique())
            
            # Skip the first value as it's dropped in one-hot encoding
            if len(unique_values) > 1:
                for val in unique_values[1:]:
                    output_features.append(f"{col}_{val}")
        
        # Add passthrough columns
        passthrough_cols = [col for col in train_df.columns 
                            if col not in log_columns + minmax_columns + 
                            standard_columns + ordinal_columns + onehot_columns]
        output_features.extend(passthrough_cols)
        
        return output_features    
    
    # Fit on training data
    train_transformed = preprocessor.fit_transform(train_df)
    
    
    # Transform validation and test data
    val_transformed = preprocessor.transform(val_df)
    test_transformed = preprocessor.transform(test_df)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    
    # Convert to DataFrames
    train_df_transformed = pd.DataFrame(train_transformed, columns=feature_names)
    val_df_transformed = pd.DataFrame(val_transformed, columns=feature_names)
    test_df_transformed = pd.DataFrame(test_transformed, columns=feature_names)
    
    # Save transformed data if path is provided
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        train_df_transformed.to_csv(os.path.join(save_path, 'train.csv'), index=False)
        val_df_transformed.to_csv(os.path.join(save_path, 'val.csv'), index=False)
        test_df_transformed.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    
    return train_df_transformed, val_df_transformed, test_df_transformed, preprocessor

# Usage Example
if __name__ == "__main__":
    # Assuming data is loaded or available as 'data'
    # train_df, val_df, test_df, preprocessor = preprocess_data(data=data, save_path='./processed_data/')
    
    # For future predictions on new data
    # def predict_preprocess(new_data, preprocessor):
    #     return pd.DataFrame(preprocessor.transform(new_data))
    pass