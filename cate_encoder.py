import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder

class CaterEncoder:
    def __init__(self):
        self.Owner_dict = {
            "UnRegistered Car": 0,
            "First": 1,
            "Second": 2,
            "Third": 3,
            "Fourth": 4,
            "4 or More": 4
        }

    def fit_transform(self, data: pd.DataFrame):
        # Model (drop)
        data.drop(columns=['Model'], inplace=True)

        # Location (binary encoding)
        self.Location_encoder= BinaryEncoder(cols=['Location'],return_df=True)
        data = self.Location_encoder.fit_transform(data)
        
        # Make (binary encoding)
        self.Make_encoder= BinaryEncoder(cols=['Make'],return_df=True)
        data = self.Make_encoder.fit_transform(data)

        # Color (one-hot encoding)
        self.Color_encoder= BinaryEncoder(cols=['Color'],return_df=True)
        data = self.Color_encoder.fit_transform(data)
        
        # Fule type (one-hot encoding)
        data = pd.get_dummies(data, columns=["Fuel Type"], drop_first=True)

        # Owner (Ordinal encoding)
        data["Owner"] = data["Owner"].map(self.Owner_dict)

        # Drivetrain (one-hot encoding)
        data = pd.get_dummies(data, columns=["Drivetrain"], drop_first=True)
        
        # Transmission (one-hot encoding)
        data = pd.get_dummies(data, columns=["Transmission"], drop_first=True)

        # Color (one-hot encoding)
        data = pd.get_dummies(data, columns=["Seller Type"], drop_first=True)

        return data

    def transform(self, data: pd.DataFrame):
        # Model (drop)
        data.drop(columns=['Model'], inplace=True)

        # Location (binary encoding)
        data = self.Location_encoder.transform(data)

        # Make (binary encoding)
        data = self.Make_encoder.transform(data)

        # Color (one-hot encoding)
        data = self.Color_encoder.transform(data)

        # Fule type (one-hot encoding)
        data = pd.get_dummies(data, columns=["Fuel Type"], drop_first=True)

        # Owner (Ordinal encoding)
        data["Owner"] = data["Owner"].map(self.Owner_dict)

        # Drivetrain (one-hot encoding)
        data = pd.get_dummies(data, columns=["Drivetrain"], drop_first=True)
        
        # Transmission (one-hot encoding)
        data = pd.get_dummies(data, columns=["Transmission"], drop_first=True)

        # Color (one-hot encoding)
        data = pd.get_dummies(data, columns=["Seller Type"], drop_first=True)

        return data

