# Car Price Prediction Project Plan

## I. Data Analysis and Feature Engineering

### 1. Data Exploration
- Load the dataset from train.csv
- Print dataset information (row count, column names)
- Display first 5 rows of training data
- Calculate descriptive statistics (mean, std, variance)
- Visualize distributions and relationships between features

### 2. Data Preprocessing
- Handle non-numeric features:
  - Extract numeric values from text+number fields (if units are consistent)
  - Convert categorical text to numerical values (ordinal encoding)
  - Apply one-hot encoding for categorical variables where appropriate
- Feature selection:
  - F-test to identify significant features
  - Forward selection/backward elimination to optimize feature set
  - Correlation analysis to understand feature relationships

### 3. Feature Visualization (Optional)
- Apply dimensionality reduction techniques:
  - PCA for feature compression while maintaining variance
  - T-SNE/UMAP for visualizing high-dimensional data relationships

## II. Model Development

### 1. Linear Regression Implementation
- Define multiple regression formulas (one per team member):
  - Standard linear: y = a₁x₁ + a₂x₂ + a₃x₃ + a₄x₄
  - Polynomial terms: y = a₁x₁² + a₂x₂ + a₃x₃² + a₄x₄
  - Mixed terms: y = a₁(x₁ + x₂) + a₃x₃² + a₄x₄
  - Interaction terms: y = a₁x₁x₂ + a₃x₃²
- Document rationale for each formula choice (in Jupyter markdown cells)

### 2. Model Training Methods
- Analytical solutions:
  - Pseudo-inverse matrix method (following Dr. Vu Huu Tiep's approach)
  - Convex optimization with different norms
- Gradient-based optimization:
  - Batch gradient descent
  - Stochastic gradient descent
  - Adam/RMSProp optimization
  - Note: Must implement these without using sklearn

### 3. Model Evaluation
- Implement error metrics (MSE/MAE) for model evaluation
- Create function to read and evaluate models on any input CSV file
- Compare performance across different regression formulas

## III. Project Management

### 1. Development Environment
- Set up Jupyter Notebook in PyCharm or VSCode
- Create GitHub repository for version control

### 2. Documentation
- Create comprehensive report including methodology and results
- Ensure proper code documentation with team member information
- Package submission as required (Report + Source folders)

### 3. Submission Preparation
- Verify code runs without errors
- Organize files according to the submission format (MSSV01_MSSV02...)
- Package as ZIP file with correct naming