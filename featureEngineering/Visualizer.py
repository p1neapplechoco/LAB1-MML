import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

class Visualizer:
    @staticmethod
    def correlation_heatmap(df, min_correlation, features_col = None, target = None, figsize=(20, 20), annot=True):
        if features_col is None:
            features_col = df.select_dtypes(include='number').columns
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
    def target_feature_scatterplots(df, target_col, features_col=None, model=None, cols_per_row=3, figsize=(5, 5)):
        if features_col is None:
            features_col = df.select_dtypes(include='number').columns
        numeric_data = df[features_col]
        
        correlations = numeric_data.corr()[target_col]
        num_cols = [col for col in numeric_data.columns if col != target_col]
        
        predictions = None
        if model is not None:
            X_model = df[num_cols]
            predictions = model.predict(X_model)
        
        num_plots = len(num_cols)
        num_rows = math.ceil(num_plots / cols_per_row)
        
        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(figsize[0] * cols_per_row, figsize[1] * num_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(num_cols):
            ax = axes[i]
            sns.scatterplot(x=numeric_data[col], y=numeric_data[target_col], ax=ax, label='Data')

            if predictions is not None:
                ax.scatter(numeric_data[col], predictions, color='red', alpha = 0.2, marker='o', label='Model Predict')

            ax.set_title(f"{col} vs {target_col} (corr = {correlations[col]:.2f})")
            ax.set_xlabel(col)
            ax.set_ylabel(target_col)
            if predictions is not None:
                ax.legend()
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def histogram_plot(df, col, figsize=(5, 5)):
        plt.figure(figsize=figsize)
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.xticks(rotation=90)
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
    
    @staticmethod
    def count_plot(df, categorical_features = None, figsize=(20, 10)):
        if categorical_features is None:
            categorical_features = df.select_dtypes(include='object').columns
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

    @staticmethod
    def result_plots(y_true, y_pred, y_true_test, y_pred_test, figsize=(15, 20)):
        fig, axs = plt.subplots(4, 2, figsize=figsize)
        
        # Row 1: Predicted vs True Plot
        # Train
        axs[0, 0].scatter(y_true, y_pred, alpha=0.6)
        m, M = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
        axs[0, 0].plot([m, M], [m, M], 'r--')
        axs[0, 0].set_title('Pred vs True - Train')
        axs[0, 0].set_xlabel('True Values')
        axs[0, 0].set_ylabel('Predicted Values')
        # Test
        axs[0, 1].scatter(y_true_test, y_pred_test, alpha=0.6)
        m, M = min(np.min(y_true_test), np.min(y_pred_test)), max(np.max(y_true_test), np.max(y_pred_test))
        axs[0, 1].plot([m, M], [m, M], 'r--')
        axs[0, 1].set_title('Pred vs True - Test')
        axs[0, 1].set_xlabel('True Values')
        axs[0, 1].set_ylabel('Predicted Values')
        
        # Tính residuals cho train và test
        resid_train = np.array(y_true) - np.array(y_pred)
        resid_test  = np.array(y_true_test) - np.array(y_pred_test)
        
        # Row 2: Residual Plot
        axs[1, 0].scatter(y_pred, resid_train, alpha=0.6)
        axs[1, 0].axhline(0, color='red', linestyle='--')
        axs[1, 0].set_title('Residual Plot - Train')
        axs[1, 0].set_xlabel('Predicted Values')
        axs[1, 0].set_ylabel('Residuals')
        
        axs[1, 1].scatter(y_pred_test, resid_test, alpha=0.6)
        axs[1, 1].axhline(0, color='red', linestyle='--')
        axs[1, 1].set_title('Residual Plot - Test')
        axs[1, 1].set_xlabel('Predicted Values')
        axs[1, 1].set_ylabel('Residuals')
        
        # Row 3: Scale-Location Plot
        axs[2, 0].scatter(y_pred, np.sqrt(np.abs(resid_train)), alpha=0.6)
        axs[2, 0].set_title('Scale-Location Plot - Train')
        axs[2, 0].set_xlabel('Predicted Values')
        axs[2, 0].set_ylabel('sqrt(|Residuals|)')
        
        axs[2, 1].scatter(y_pred_test, np.sqrt(np.abs(resid_test)), alpha=0.6)
        axs[2, 1].set_title('Scale-Location Plot - Test')
        axs[2, 1].set_xlabel('Predicted Values')
        axs[2, 1].set_ylabel('sqrt(|Residuals|)')
        
        # Row 4: Q-Q Plot
        stats.probplot(resid_train, dist="norm", plot=axs[3, 0])
        axs[3, 0].set_title('Q-Q Plot - Train')
        axs[3, 0].set_xlabel('Theoretical Quantiles')
        axs[3, 0].set_ylabel('Sample Quantiles')
        
        stats.probplot(resid_test, dist="norm", plot=axs[3, 1])
        axs[3, 1].set_title('Q-Q Plot - Test')
        axs[3, 1].set_xlabel('Theoretical Quantiles')
        axs[3, 1].set_ylabel('Sample Quantiles')
        
        plt.tight_layout()
        plt.show()
            