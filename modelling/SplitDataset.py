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
    
    def get_train_data(self):
        return self.train_df
    
    def get_val_data(self):
        return self.val_df
    
    def get_test_data(self):
        return self.test_df