# LAB1-MML

This repository contains the solution for Lab 1 of the "Mathematics for Artificial Intelligence" course (Toán cho Trí tuệ Nhân Tạo), focusing on predicting used car prices using Linear Regression.

## Dataset
The project uses `train.csv` dataset containing:
- Features: Make, Model, Year, Kilometer, Fuel Type, Transmission, Location, Color, Owner, Seller Type, Engine, Max Power, Max Torque, DriveTrain, Length, Width, Height, Seating Capacity, Fuel Tank Capacity
- Target variable: Selling price (Y)

## Methodology
### Data Preprocessing
1. Feature selection from the dataset
2. Handling non-numeric data:
   - Extracting numerical values from mixed text/number columns
   - Encoding categorical features numerically
3. Optional preprocessing steps:
   - Missing value imputation
   - Feature scaling

### Feature Engineering
- Creating derived features and interaction terms

### Model Implementation
- Building Linear Regression models from scratch (without sklearn)
- Implementing multiple regression formulas as required. (Standard, Polynomial, Interaction, Mixed)
- Training models on preprocessed data
- Evaluating performance using MSE/MAE

### Prediction
- Code to predict prices on new CSV data
- Model evaluation on unseen data

## Repository Structure
- `data/train.csv`: Contains dataset files
- `preprocessed_data/`: Contains preprocessed data (after running `main.ipynb`)
- `preprocessor/sklearn_preprocessor.py`: Data preprocessing scripts
- `featureEngineering/`: Feature transformation code
- `modelling/`: Linear regression implementation and Loss function defined
- `main.ipynb`: Main notebook with analysis and modeling


## Acknowledgement
### Team Members:
Mai Duc Minh Huy (SaberToaster) <br>
Dinh Duc Anh Khoa <br>
Nguyen Thien An <br>
Luu Thuong Hong
### Supervisors:
TS. Can Tran Thanh Trung <br>
Mr. Nguyen Ngoc Toan
