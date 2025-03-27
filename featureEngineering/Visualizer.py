import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

class Visualizer:
    @staticmethod
    def correlation_heatmap(df, features_col, min_correlation, target = None, figsize=(20, 20), annot=True):
        numeric_data = df[features_col]
        corr_matrix = numeric_data.corr()

        mask = abs(corr_matrix) < min_correlation

        if(target is None):
            target = features_col[0]
        filtered_corr = corr_matrix.mask(mask)
        sort_order = filtered_corr[target].abs().sort_values(ascending=False).index
        filtered_corr = filtered_corr.loc[sort_order, sort_order]

        plt.figure(figsize=figsize)
        sns.heatmap(filtered_corr, cmap="coolwarm", vmin=-1, vmax=1, annot=annot)
        plt.title(f"Feature Correlation Heatmap (|corr| >= {min_correlation})")
        plt.show()
        plt.close()


    @staticmethod
    def target_feature_scatterplots(df, features_col, target_col, cols_per_row=3, figsize=(5, 5)):
        features_col.append(target_col)
        features_col = list(set(features_col))
        numeric_data = df[features_col]
        
        correlations = numeric_data.corr()[target_col]
        num_cols = [col for col in numeric_data.columns if col != target_col]

        num_plots = len(num_cols)
        num_rows = math.ceil(num_plots / cols_per_row)

        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(figsize[0] * cols_per_row, figsize[1] * num_rows))
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            sns.scatterplot(x=numeric_data[col], y=numeric_data[target_col], ax=axes[i])
            axes[i].set_title(f"{col} vs {target_col} (corr = {correlations[col]:.2f})")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(target_col)

        # Xóa biểu đồ thừa
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def residual_plot(y_true, y_pred, figsize=(5, 5)):
        residuals = y_true - y_pred
        plt.figure(figsize=figsize)
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title("Residual Plot")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.show()
        plt.close()

    @staticmethod
    def qq_plot(y_true, y_pred, figsize=(5, 5)):
        residuals = y_true - y_pred
        plt.figure(figsize=figsize)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.show()
        plt.close()
    
    @staticmethod
    def scale_location_plot(y_true, y_pred, figsize=(5, 5)):
        residuals = y_true - y_pred
        sqrt_abs_residuals = residuals.abs().apply(math.sqrt)
        plt.figure(figsize=figsize)
        sns.scatterplot(x=y_pred, y=sqrt_abs_residuals)
        plt.title("Scale-Location Plot")
        plt.xlabel("Predicted Values")
        plt.ylabel("sqrt(|Residuals|)")
        plt.show()
        plt.close()

    @staticmethod
    def boxplot(df, col, figsize=(5, 5)):
        plt.figure(figsize=figsize)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()
        plt.close()
    
    @staticmethod
    def histogram_plot(df, col, figsize=(5, 5)):
        plt.figure(figsize=figsize)
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.show()
        plt.close()
    
    @staticmethod
    def distribution_plot(df, numerical_features, figsize=(25,25)):
        plt.figure(figsize=figsize)
        for feature in numerical_features:
            plt.subplot(5, 5, numerical_features.index(feature) + 1)
            sns.histplot(data=df[feature], bins=20, kde=True)
            plt.title(feature)
        plt.tight_layout()
        plt.show()
    
    def count_plot(df, categorical_features, figsize=(20, 10)):
        num_plots = len(categorical_features)
        ncols = 3
        nrows = int(np.ceil(num_plots / ncols))
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
        
        for i, column in enumerate(categorical_features):
            ax = axes[i]
            sns.countplot(x=df[column], data=df, palette='bright', ax=ax, saturation=0.95)
            for container in ax.containers:
                ax.bar_label(container, color='black', size=10)
            ax.set_title(f'Count Plot of {column.capitalize()}')
            ax.set_xlabel(column.capitalize())
            ax.set_ylabel('Count')
        
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()