# Dự đoán giá xe cũ sử dụng mô hình hồi quy tuyến tính

## Vấn đề:

Trong bài lab này, chúng ta sẽ xây dựng mô hình dự đoán giá xe hơi cũ dựa trên các đặc điểm của xe như năm sản xuất, số km đã đi, thông số động cơ, v.v.

## Mục tiêu:
- Phân tích các yếu tố ảnh hưởng đến giá bán của xe hơi cũ
- Xây dựng các mô hình hồi quy khác nhau để dự đoán giá xe
- So sánh hiệu suất của các mô hình để chọn ra mô hình tốt nhất

## Cách tiếp cận:
1. Đọc và tìm hiểu dữ liệu
2. Tiền xử lý dữ liệu
3. Phân tích dữ liệu và lựa chọn đặc trưng
4. Xây dựng các mô hình hồi quy tuyến tính
5. Đánh giá và so sánh các mô hình
6. Dự đoán trên dữ liệu kiểm tra

## Cài đặt thư viện cần thiết

```python
# %pip install numpy
# %pip install pandas
# %pip install matplotlib
# %pip install seaborn
# %pip install scipy
```

## I. Đọc dữ liệu

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

#### Insights:
- Tập dữ liệu gồm 1647 mẫu với 20 đặc trưng (features)
- Các đặc trưng bao gồm thông tin về nhãn hiệu (Make), mẫu xe (Model), giá (Price), năm sản xuất (Year), số km đã đi (Kilometer), v.v.
- Có một số giá trị NaN trong dữ liệu, đặc biệt là trong các trường thông số kỹ thuật (Engine, Max Power, Max Torque)

## II. Tiền xử lý dữ liệu

### 1. Làm sạch dữ liệu

```python
def clean_data(data):
    data["Engine"] = data["Engine"].str.replace(' cc', '', regex=False).astype(float)

    data[['Max Power BHP', 'Max Power RPM']] = data['Max Power'].str.extract(r'(\d+)\s*bhp\s*@\s*(\d+)\s*rpm')
    data['Max Power BHP'] = pd.to_numeric(data['Max Power BHP'], errors='coerce')
    data['Max Power RPM'] = pd.to_numeric(data['Max Power RPM'], errors='coerce')

    data[['Max Torque Nm', 'Max Torque RPM']] = data['Max Torque'].str.extract(r'(\d+)\s*Nm\s*@\s*(\d+)\s*rpm')
    data['Max Torque Nm'] = pd.to_numeric(data['Max Torque Nm'], errors='coerce')
    data['Max Torque RPM'] = pd.to_numeric(data['Max Torque RPM'], errors='coerce')

    rare_fuel_types = ['Electric', 'LPG', 'Hybrid', 'CNG + CNG', 'Petrol + LPG']
    data["Fuel Type"] = data["Fuel Type"].apply(lambda x: 'Others' if x in rare_fuel_types else x)

    drop_columns = ['Make', 'Model', 'Location', 'Color', 'Max Power', 'Max Torque']
    data.drop(drop_columns, axis=1, inplace=True)
    
    return data
```

```python
data = clean_data(data)
```

#### Insights:
- Chuyển đổi các giá trị từ dạng text sang dạng số học:
  - Engine: Loại bỏ "cc" và chuyển thành kiểu float
  - Max Power: Tách thành công suất (BHP) và vòng tua (RPM)
  - Max Torque: Tách thành mô-men xoắn (Nm) và vòng tua (RPM)
- Nhóm các loại nhiên liệu hiếm vào một nhóm 'Others'
- Loại bỏ các cột ít liên quan hoặc có quá nhiều giá trị khác nhau như Make, Model, Location, Color

### 2. Phân chia và tiền xử lý dữ liệu

```python
# Apply splitting and preprocessing
train_df, test_df, preprocessor, feature_names = preprocess_data(
    data=data,
    save_path='./processed_data/',
    train_ratio=0.7,
    test_ratio=0.3,
)
```

```python
# Check if there are any columns with object type in the transformed dataframes
print("Train DataFrame Object Types:", train_df.select_dtypes(include=['object']).columns.tolist())
print("Test DataFrame Object Types:", test_df.select_dtypes(include=['object']).columns.tolist())

# If there are object columns, convert them to numeric
for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col])

# Verify the conversion worked
print("\nAfter conversion:")
print("Train DataFrame Types:", train_df.dtypes.value_counts())
print("Test DataFrame Types:", test_df.dtypes.value_counts())
```

#### Insights:
- Dữ liệu được chia thành tập huấn luyện (70%) và tập kiểm tra (30%)
- Tất cả các cột đã được chuyển đổi sang kiểu số học (float64)
- Các biến phân loại (categorical) đã được mã hóa one-hot

## III. Mô hình hóa

### 1. Chọn các cột đặc trưng

#### Phân tích tương quan

```python
Visualizer.correlation_heatmap(train_df, min_correlation=0.3, target='Price')
```

```python
from modelling.Model import StandardRegression, PolynomialRegression

oneHotCols = ['Drivetrain', 'Fuel Type', 'Seller Type', 'Transmission']
model_type = StandardRegression
model = model_type()

train = model.transform_features(train_df)
test = model.transform_features(test_df)
corr = train.corr()
sort_order = corr['Price'].abs().sort_values(ascending=False).index
sorted_corr = corr.loc[sort_order, sort_order]
subcols = sorted_corr.index.tolist()

# Chọn thử 14 cột có abs correlation >= 0.3 sau khi visualize
subcols = subcols[:8]
subtract = ['Price']
subcols = list(set(subcols) - set(subtract))
add = []
subcols = subcols + add
subcols
```

#### Insights:
- Từ biểu đồ nhiệt tương quan (correlation heatmap), chúng ta có thể thấy các đặc trưng có mối tương quan mạnh với giá xe (Price)
- Các đặc trưng được chọn dựa trên tương quan cao với biến mục tiêu (có độ lớn tương quan ≥ 0.3)
- Danh sách các đặc trưng quan trọng bao gồm: dung tích bình nhiên liệu, loại hộp số, hệ dẫn động, chiều rộng, dung tích động cơ, chiều dài, và công suất động cơ

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
def model_testing(subcols=None, linear=True, deg=2, auto_selection = True, 
            model_type=None, plot=True):
    if subcols is None:
        subcols = train_df.drop('Price', axis=1).columns.to_list()
    
    if linear:
        model = StandardRegression()
    else:
        model = model_type(degree=deg)

    train = model.transform_features(train_df.drop('Price', axis=1))
    test = model.transform_features(test_df.drop('Price', axis=1))
    train['Price'] = train_df['Price']
    test['Price'] = test_df['Price']

    # Forward selection
    if auto_selection:
        subcols, r2 = FeatureSelection.backward_elimination_mae(
            train[subcols + ['Price']],
            oneHotCols=oneHotCols,
            target='Price',
            model = model,
        )
        if plot:
            print('Best:', r2)
            print('Eliminated columns:', set(train.columns.to_list()) - set(subcols))
    
    print('Chosed columns: ', subcols)
    print('Total columns: ', len(subcols))
    if(plot):
        model.fit(train[subcols], train['Price'])
        y_train = train['Price']
        y_test = test['Price']
        y_train_pred = model.predict(train[subcols])
        y_test_pred = model.predict(test[subcols])

        # Evaluate after selecting features
        print('Train r^2: ', r2_score(y_train, y_train_pred))
        print('Train MAE: ', f"{mean_absolute_error(y_train, y_train_pred):.3e}")
        print('Train MSE: ', f"{mean_squared_error(y_train, y_train_pred):.3e}")
        print('Train RMSE: ', f"{np.sqrt(mean_squared_error(y_train, y_train_pred)):.3e}")

        print('Test r^2: ', r2_score(y_test, y_test_pred))
        print('Test MAE: ', f"{mean_absolute_error(y_test, y_test_pred):.3e}")
        print('Test MSE: ', f"{mean_squared_error(y_test, y_test_pred):.3e}")
        print('Test RMSE: ', f"{np.sqrt(mean_squared_error(y_test, y_test_pred)):.3e}")

        # diagnostic plots
        Visualizer.pred_vs_true_plot(y_train, y_train_pred, y_test, y_test_pred, figsize=(15, 5))
        Visualizer.residual_plot([y_train, y_train_pred], [y_test, y_test_pred])
        Visualizer.qq_plot((y_train, y_train_pred), (y_test, y_test_pred))
        Visualizer.scale_location_plot((y_train, y_train_pred), (y_test, y_test_pred))
    return subcols
```

### 2. So sánh các mô hình

#### Hồi quy tuyến tính (sử dụng tất cả đặc trưng)

```python
model_testing(auto_selection=False);
```

#### Insights:
- Mô hình sử dụng tất cả 21 đặc trưng
- R² cho tập huấn luyện: 0.825, tập kiểm tra: 0.893
- RMSE cho tập huấn luyện: 1.058e+06, tập kiểm tra: 6.470e+05
- Mô hình có khả năng dự đoán tốt trên cả tập huấn luyện và kiểm tra

#### Hồi quy tuyến tính (chỉ với các đặc trưng có |tương quan| > 0.3)

```python
model_testing(subcols=subcols, linear=True, auto_selection=False);
```

#### Insights:
- Mô hình này chỉ sử dụng 7 đặc trưng: Fuel Tank Capacity, Transmission_Manual, Drivetrain_FWD, Width, Engine, Length, Max Power BHP
- R² cho tập huấn luyện: 0.667, tập kiểm tra: 0.720
- RMSE cao hơn so với mô hình sử dụng tất cả đặc trưng
- Đây là một mô hình đơn giản hơn nhưng vẫn có khả năng dự đoán khá tốt

#### Hồi quy tuyến tính với tự động lựa chọn đặc trưng

```python
model_testing(auto_selection=True);
```

#### Insights:
- Thông qua quá trình lựa chọn đặc trưng tự động (backword elimination), mô hình đã loại bỏ 'Owner' và 'Price'
- R² cho tập huấn luyện: 0.829, tập kiểm tra: 0.892
- Mô hình này có hiệu suất gần tương đương với mô hình sử dụng tất cả đặc trưng nhưng đơn giản hơn
- Điều này cho thấy 'Owner' không ảnh hưởng nhiều đến dự đoán giá xe

#### Hồi quy đa thức (Polynomial Regression) bậc 2

```python
model_testing(linear=False, 
        deg=2,
        auto_selection=True, 
        model_type=PolynomialRegression);
```

#### Insights:
- Mô hình hồi quy đa thức bậc 2 đã loại bỏ nhiều đặc trưng không cần thiết (các biến bậc 2)
- R² cho tập huấn luyện: 0.824, tập kiểm tra: 0.873
- Mô hình này có hiệu suất tốt nhưng không cải thiện đáng kể so với hồi quy tuyến tính
- Phân phối phần dư (residuals) khá đồng đều, cho thấy mô hình phù hợp với dữ liệu

#### Hồi quy hỗn hợp (Mixed Regression) với các đặc trưng tương tác

```python
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)

from modelling.Model import MixedRegression
model_testing(linear=False,
        deg=2,
        auto_selection=True, 
        model_type=MixedRegression);
```

#### Insights:
- Mô hình hồi quy hỗn hợp tạo ra nhiều đặc trưng tương tác giữa các đặc trưng gốc
- R² cho tập huấn luyện rất cao (0.957) nhưng R² cho tập kiểm tra thấp hơn (0.752)
- Sự chênh lệch lớn giữa hiệu suất trên tập huấn luyện và kiểm tra cho thấy mô hình bị overfitting
- Mặc dù mô hình này phức tạp nhất, nó không phải là mô hình tốt nhất để dự đoán giá xe do vấn đề overfitting

### 3. Chọn mô hình tổng quát nhất cho dự đoán

```python
# all in
subcols = model_testing(auto_selection=True, plot=False)
model.fit(train_df[subcols], train_df['Price'])
```

#### Insights:
- Sau khi so sánh các mô hình, chúng ta chọn mô hình hồi quy tuyến tính với lựa chọn đặc trưng tự động
- Mô hình này cung cấp sự cân bằng tốt giữa độ chính xác và độ phức tạp
- Các đặc trưng được chọn bao gồm các thông số kỹ thuật quan trọng của xe như năm sản xuất, số km đã đi, công suất động cơ, kích thước xe, v.v.

## IV. Dự đoán trên tập dữ liệu mới

```python
data_path = 'test.csv'
test_data = pd.read_csv(data_path)
test_data = clean_data(test_data)
# test_data.dropna(inplace=True)
y = test_data['Price']

test_data = preprocessor.transform(test_data)
test_data = pd.DataFrame(test_data, columns=feature_names)

model.predict(test_data[subcols])

y_pred = model.predict(test_data[subcols])
print('Test r^2: ', r2_score(y, y_pred))
print('Test MAE: ', f"{mean_absolute_error(y, y_pred):.3e}")
print('Test MSE: ', f"{mean_squared_error(y, y_pred):.3e}")
```

## Kết luận:

1. **Mô hình tốt nhất**: Hồi quy tuyến tính với lựa chọn đặc trưng tự động là mô hình tốt nhất cho bài toán dự đoán giá xe cũ
2. **Đặc trưng quan trọng**: Các đặc trưng quan trọng nhất bao gồm năm sản xuất, số km đã đi, công suất động cơ, kích thước xe, loại nhiên liệu, và hệ dẫn động
3. **Hiệu suất mô hình**: Mô hình có R² xấp xỉ 0.84 trên tập dữ liệu kiểm tra, cho thấy khả năng dự đoán tốt
4. **Hạn chế**: Mô hình hồi quy đa thức và hỗn hợp không cải thiện đáng kể hiệu suất, thậm chí mô hình hỗn hợp còn bị overfitting

Mô hình hồi quy tuyến tính với lựa chọn đặc trưng tự động là sự lựa chọn phù hợp nhất cho bài toán này vì nó cung cấp sự cân bằng tốt giữa độ chính xác và độ phức tạp.