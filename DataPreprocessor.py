import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from collections import Counter

class DataFiller:
    def __init__(self, df):
        self.df = df

    def fill_mean(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
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

                for col in self.df.columns:
                    if pd.isnull(self.df.at[i, col]):
                        neighbor_values = [neighbor[col] for neighbor in nearest_neighbors]
                        neighbor_values = [val for val in neighbor_values if not pd.isnull(val)]

                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            self.df.at[i, col] = np.mean(neighbor_values)
                        else:
                            self.df.at[i, col] = Counter(neighbor_values).most_common(1)[0][0]

        return self.df

class DataEncoder:
    def __init__(self, df):
        self.df = df

    def one_hot_encode(self, columns=None, threshold=0):
        for col in columns:
            rare = self.df.value_counts()[lambda x: x < threshold].index
            self.df[col] = self.df[col].apply(lambda x: x if x not in rare else 'Other')
            self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
        return self.df

    def ordinal_encode(self, col=None, to: dict=None):
        self.df[col] = self.df[col].map(to)
        return self.df

class DataScaler:
    def __init__(self, df):
        self.df = df

    def log_norm(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if is_numeric_dtype(self.df[col]):
                self.df[col] = np.log(self.df[col] + 1)
        return self.df

    def minmax_norm(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if is_numeric_dtype(self.df[col]):
                self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())

        return self.df

    def standard_norm(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if is_numeric_dtype(self.df[col]):
                self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
        return self.df

    def robust_norm(self, columns=None, focus=0.5):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if is_numeric_dtype(self.df[col]):
                self.df[col] = (self.df[col] - self.df[col].median()) / (self.df[col].quantile(1 - focus / 2) - self.df[col].quantile(focus / 2))
        return self.df

class DataPreprocessor(DataFiller, DataEncoder, DataScaler):
    def __init__(self, df):
        DataFiller.__init__(self, df)
        DataEncoder.__init__(self, df)
        DataScaler.__init__(self, df)