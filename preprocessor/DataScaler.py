import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

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