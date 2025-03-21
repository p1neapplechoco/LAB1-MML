import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from collections import Counter, defaultdict


class DataFiller:
    def __init__(self, df):
        self.df = df

        self.mean = defaultdict(float)
        self.median = defaultdict(float)
        self.std = defaultdict(float)
        self.mode = defaultdict(int)
        self.weights = defaultdict(float)

    def fill_mean(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            self.mean[col] = self.df[col].mean()
            if is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self.df

    def fill_median(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
        return self.df

    def fill_std(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].std())
        return self.df

    def fill_mode(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self.df

    def fill_knn(self, columns=None, k=3):
        def euclidean_distance(row1, row2):
            numeric_mask = pd.api.types.is_numeric_dtype(row1) and pd.api.types.is_numeric_dtype(row2)

            if numeric_mask:
                mask = ~np.isnan(row1) & ~np.isnan(row2)
                if np.any(mask):
                    return np.sqrt(np.sum((row1[mask] - row2[mask]) ** 2))
                else:
                    return np.inf
            else:
                return np.inf

        for i, row in self.df.iterrows():
            if row.isnull().any():
                distances = []

                for j, other_row in self.df.iterrows():
                    if not other_row.isnull().any():
                        dist = euclidean_distance(row.values, other_row.values)
                        distances.append((dist, j))

                distances = sorted(distances, key=lambda x: x[0])
                nearest_neighbors = [self.df.iloc[j] for dist, j in distances[:k]]

                for col in columns:
                    if pd.isnull(self.df.at[i, col]):
                        neighbor_values = [neighbor[col] for neighbor in nearest_neighbors]
                        neighbor_values = [val for val in neighbor_values if not pd.isnull(val)]

                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            self.df.at[i, col] = np.mean(neighbor_values)
                        else:
                            self.df.at[i, col] = Counter(neighbor_values).most_common(1)[0][0]

        return self.df
