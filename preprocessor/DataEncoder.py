import pandas as pd
import numpy as np

class DataEncoder:
    def __init__(self, df):
        self.df = df

    def one_hot_encode(self, columns=None, threshold=0):
        for col in columns:
            if threshold:
                rare = self.df.value_counts()[lambda x: x < threshold].index
                self.df[col] = self.df[col].apply(lambda x: x if x not in rare else 'Other')
            self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
        return self.df

    def ordinal_encode(self, col=None, to: dict=None):
        self.df[col] = self.df[col].map(to)
        return self.df