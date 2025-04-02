import pandas as pd
import numpy as np
import math

from collections import defaultdict
from pandas.api.types import is_numeric_dtype

class DataScaler:
    def __init__(self, df):
        self.df = df

        self.log = defaultdict(float)
        self.minmax = defaultdict(tuple)
        self.std = defaultdict(tuple)
        self.robust = defaultdict(tuple)

    def log_norm(self, columns=None, log=np.e):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            self.log[col] = log
            if is_numeric_dtype(self.df[col]):
                 self.df[col] = np.emath.logn(log, self.df[col] + 1)
        return self.df

    def minmax_norm(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            self.minmax[col] = (self.df[col].min(), self.df[col].max() - self.df[col].min())
            if is_numeric_dtype(self.df[col]):
                self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())

        return self.df

    def standard_norm(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            self.std[col] = (self.df[col].mean(), self.df[col].std())
            if is_numeric_dtype(self.df[col]):
                self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
        return self.df

    def robust_norm(self, columns=None, focus=0.5):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            self.robust[col] = (self.df[col].median(), self.df[col].quantile(1 - focus / 2) - self.df[col].quantile(focus / 2))
            if is_numeric_dtype(self.df[col]):
                self.df[col] = (self.df[col] - self.df[col].median()) / (self.df[col].quantile(1 - focus / 2) - self.df[col].quantile(focus / 2))
        return self.df

    def denormalize(self, y_scaled, col=None):
        a, b = 0, 0

        if col in self.log.keys():
            return self.log[col] ** y_scaled - 1

        elif col in self.minmax.keys():
            a, b = self.minmax[col]

        elif col in self.std.keys():
            a, b = self.std[col]

        elif col in self.robust.keys():
            a, b = self.robust[col]

        else:
            return y_scaled

        return b * y_scaled + a