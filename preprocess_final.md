# Installing required libraries


```python
# %pip install numpy
# %pip install pandas
# %pip install matplotlib
# %pip install seaborn
# %pip install scipy
```


```python
from preprocessor.sklearn_preprocessor import preprocess_data
from featureEngineering.Visualizer import Visualizer
from featureEngineering.FeatureSelection import FeatureSelection

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```


```python
path = './data/'
data = pd.read_csv(path + 'train.csv')
```


```python
print(f'Số lượng dòng của data: {len(data)}')
print(data.columns)
```


```python
data.head()
```


```python
data.info()
```

# I. Preprocessing Data

## 1. Cleaning data



```python
data.dropna(how='all')
data.drop_duplicates()

data["Engine"] = data["Engine"].str.replace(' cc', '', regex=False).astype(float)

data[['Max Power BHP', 'Max Power RPM']] = data['Max Power'].str.extract(r'(\d+)\s*bhp\s*@\s*(\d+)\s*rpm')
data['Max Power BHP'] = pd.to_numeric(data['Max Power BHP'], errors='coerce')
data['Max Power RPM'] = pd.to_numeric(data['Max Power RPM'], errors='coerce')

data[['Max Torque Nm', 'Max Torque RPM']] = data['Max Torque'].str.extract(r'(\d+)\s*Nm\s*@\s*(\d+)\s*rpm')
data['Max Torque Nm'] = pd.to_numeric(data['Max Torque Nm'], errors='coerce')
data['Max Torque RPM'] = pd.to_numeric(data['Max Torque RPM'], errors='coerce')

data = data.drop('Max Power', axis=1)
data = data.drop('Max Torque', axis=1)
```

spltio



```python
data["Drivetrain"].value_counts()
```


```python
data["Fuel Type"].value_counts()
```


```python
# Get the unique values before making changes
print("Unique values before:", data["Fuel Type"].unique())

# Replace less common fuel types with 'Others'
rare_fuel_types = ['Electric', 'LPG', 'Hybrid', 'CNG + CNG', 'Petrol + LPG']
data["Fuel Type"] = data["Fuel Type"].apply(lambda x: 'Others' if x in rare_fuel_types else x)

# Check the unique values after making changes
print("Unique values after:", data["Fuel Type"].unique())
```

## 2. Adding Interaction Terms

## 3. Splitting and preprocessing


```python
data.info()
```


```python
# Apply preprocessing
train_df, val_df, test_df, preprocessor = preprocess_data(
    data=data,
    save_path='./processed_data/',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
)
```


```python
Visualizer.target_feature_scatterplots(data, 'Price')
```


```python
### Insert feature engineer code 
### For simplicity, just choose all features, drop Make, Model and standard hypothesis model
drop_columns = ['Make', 'Model', 'Location', 'Color']
train_df = train_df.drop(drop_columns, axis=1)
val_df = val_df.drop(drop_columns, axis=1)
test_df = test_df.drop(drop_columns, axis=1)

# Check if there are any columns with object type in the transformed dataframes
print("Train DataFrame Object Types:", train_df.select_dtypes(include=['object']).columns.tolist())
print("Val DataFrame Object Types:", val_df.select_dtypes(include=['object']).columns.tolist())
print("Test DataFrame Object Types:", test_df.select_dtypes(include=['object']).columns.tolist())

# If there are object columns, convert them to numeric
for df in [train_df, val_df, test_df]:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col])

# Verify the conversion worked
print("\nAfter conversion:")
print("Train DataFrame Types:", train_df.dtypes.value_counts())
print("Val DataFrame Types:", val_df.dtypes.value_counts())
print("Test DataFrame Types:", test_df.dtypes.value_counts())
```

# II. Modeling


```python
oneHotCols = ['Drivetrain', 'Fuel Type', 'Seller Type', 'Transmission']
```


```python
from modelling.Model import StandardRegression

model_type = StandardRegression
model = model_type()
train_df = model.transform_features(train_df)
val_df = model.transform_features(val_df)
test_df = model.transform_features(test_df)

model.fit(train_df.drop('Price', axis=1), train_df['Price'])
print('Train', model.score_log(train_df.drop('Price', axis=1), np.exp(train_df['Price'])))
print('Val', model.score_log(val_df.drop('Price', axis=1), np.exp(val_df['Price'])))
print('Test', model.score_log(test_df.drop('Price', axis=1), np.exp(test_df['Price'])))
```


```python
subcols = train_df.drop(columns='Price').columns.tolist()
```


```python
# subcols, mse = FeatureSelection.forward_selection_mse(
#     train_df[subcols + ['Price']],
#     oneHotCols=oneHotCols,
#     target='Price',
#     model_type=model_type
# )
# print(subcols)
# print('mse on transformed target:', mse)
# set(train_df.columns.to_list()) - set(subcols)
```


```python
# subcols, mae = FeatureSelection.forward_selection_mae(
#     train_df[subcols + ['Price']],
#     oneHotCols=oneHotCols, 
#     target='Price',
#     model_type=model_type
# )
# print(subcols)
# print('mae on transformed target:', mae)
# set(train_df.columns.to_list()) - set(subcols)
```


```python
# subcols, r2 = FeatureSelection.forward_selection_r2(
#     train_df[subcols + ['Price']],
#     oneHotCols=oneHotCols,
#     target='Price',
#     model_type=model_type
# )
# print(subcols)
# print('Adjusted log r^2 on transformed target:', r2)
# set(train_df.columns.to_list()) - set(subcols)
```


```python
# if len(subcols) < 20:
#     subcols, r2 = FeatureSelection.subset_selection(
#         train_df[subcols + ['Price']],
#         val_df[subcols + ['Price']],
#         target='Price',
#         model_type=model_type,
#         oneHotCols=oneHotCols,
#         min_feat=10
#     )
#     print(subcols)
#     print('Adjusted log r^2 on transformed target:', r2)
# set(train_df.columns.to_list()) - set(subcols)
```


```python
eliminate_features = []
subcols = list(set(subcols) - set(eliminate_features))
```


```python
model.fit(train_df[subcols], train_df['Price'])
y_train = np.exp(train_df['Price'])
y_val = np.exp(val_df['Price'])
y_test = np.exp(test_df['Price'])
y_train_pred = np.exp(model.predict(train_df[subcols]))
y_val_pred = np.exp(model.predict(val_df[subcols]))
y_test_pred = np.exp(model.predict(test_df[subcols]))
```


```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print('Train r^2: ', r2_score(y_train, y_train_pred))
print('Train MAE: ', f"{mean_absolute_error(y_train, y_train_pred):.3e}")
print('Train MSE: ', f"{mean_squared_error(y_train, y_train_pred):.3e}")

print('Val r^2: ', r2_score(y_val, y_val_pred))
print('Val MAE: ', f"{mean_absolute_error(y_val, y_val_pred):.3e}")
print('Val MSE: ', f"{mean_squared_error(y_val, y_val_pred):.3e}")

print('Test r^2: ', r2_score(y_test, y_test_pred))
print('Test MAE: ', f"{mean_absolute_error(y_test, y_test_pred):.3e}")
print('Test MSE: ', f"{mean_squared_error(y_test, y_test_pred):.3e}")
```


```python
Visualizer.residual_plot([y_train, y_train_pred], [y_val, y_val_pred], [y_test, y_test_pred])
```


```python
Visualizer.qq_plot((y_train, y_train_pred), (y_val, y_val_pred), (y_test, y_test_pred))
```


```python
Visualizer.scale_location_plot((y_train, y_train_pred), (y_val, y_val_pred), (y_test, y_test_pred))
```
