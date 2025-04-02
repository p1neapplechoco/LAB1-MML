import numpy as np
import pandas as pd

class SplitDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def split_data(self, train_ratio=0.6, val_ratio=0.2, random_seed=42):
        """Split the dataset into train, validation, and test sets."""
        np.random.seed(random_seed)
        n_samples = len(self.dataset)
        indices = np.random.permutation(n_samples)

        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        self.train_df = self.dataset.iloc[train_indices].reset_index(drop=True)
        self.val_df = self.dataset.iloc[val_indices].reset_index(drop=True)
        self.test_df = self.dataset.iloc[test_indices].reset_index(drop=True)
        
        return self.train_df, self.val_df, self.test_df
    
    def split_data_advanced(self, train_ratio=0.6, val_ratio=0.2, random_seed=42):
        """Split the dataset into train, validation, and test sets."""
        np.random.seed(random_seed)

        # Separate rows with no missing values and rows with missing values
        full_samples = self.dataset.dropna()
        missing_samples = self.dataset[self.dataset.isnull().any(axis=1)]

        # Calculate the portion of full samples vs missing value samples
        total_samples = len(self.dataset)
        full_samples_ratio = len(full_samples) / total_samples
        missing_samples_ratio = len(missing_samples) / total_samples

        print(f"Full samples ratio: {full_samples_ratio:.2f}")
        print(f"Missing samples ratio: {missing_samples_ratio:.2f}")

        # Split full samples into train, validation, and test sets
        n_full_samples = len(full_samples)
        indices = np.random.permutation(n_full_samples)

        train_size = int(n_full_samples * train_ratio)
        val_size = int(n_full_samples * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        self.train_df = full_samples.iloc[train_indices].reset_index(drop=True)
        self.val_df = full_samples.iloc[val_indices].reset_index(drop=True)
        self.test_df = full_samples.iloc[test_indices].reset_index(drop=True)

        # Optionally, handle missing samples (e.g., imputation or separate processing)
        # For now, we just print their count
        print(f"Missing samples count: {len(missing_samples)}")

        return self.train_df, self.val_df, self.test_df

    def split_data_dropna(self, train_ratio=0.6, val_ratio=0.2, random_seed=42, subsets=[]):
        """Split the dataset into train, validation, and test sets after dropping missing values."""
        np.random.seed(random_seed)

        # Drop missing values
        if (len(subsets) == 0):
            self.dataset.dropna(inplace=True)
        else:
            self.dataset.dropna(subset=subsets, inplace=True)

        # Split the dataset
        self.split_data(train_ratio, val_ratio, random_seed)

        return self.train_df, self.val_df, self.test_df

    def get_train_data(self):
        return self.train_df
    
    def get_val_data(self):
        return self.val_df
    
    def get_test_data(self):
        return self.test_df