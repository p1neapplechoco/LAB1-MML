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
    def residual_plot(train_pair, val_pair=None, test_pair=None, figsize=(15, 5)):
        pairs = [("Train", train_pair)]
        if val_pair is not None:
            pairs.append(("Validation", val_pair))
        if test_pair is not None:
            pairs.append(("Test", test_pair))
            
        n_plots = len(pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0], figsize[1]))
        if n_plots == 1:
            axes = [axes]
        
        for ax, (label, (y_true, y_pred)) in zip(axes, pairs):
            residuals = np.array(y_true) - np.array(y_pred)
            sns.scatterplot(x=y_pred, y=residuals, ax=ax)
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_title(f"Residual Plot - {label}")
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
        
        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def qq_plot(train_pair, val_pair=None, test_pair=None, figsize=(15, 5)):
        pairs = [("Train", train_pair)]
        if val_pair is not None:
            pairs.append(("Validation", val_pair))
        if test_pair is not None:
            pairs.append(("Test", test_pair))
        
        n_plots = len(pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        for ax, (label, (y_true, y_pred)) in zip(axes, pairs):
            residuals = np.array(y_true) - np.array(y_pred)
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot - {label}")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            
        plt.tight_layout()
        plt.show()
        plt.close()
    
    @staticmethod
    def scale_location_plot(train_pair, val_pair=None, test_pair=None, figsize=(15, 5)):
        pairs = [("Train", train_pair)]
        if val_pair is not None:
            pairs.append(("Validation", val_pair))
        if test_pair is not None:
            pairs.append(("Test", test_pair))
        
        n_plots = len(pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
            
        for ax, (label, (y_true, y_pred)) in zip(axes, pairs):
            residuals = np.array(y_true) - np.array(y_pred)
            sqrt_abs_residuals = np.sqrt(np.abs(residuals))
            sns.scatterplot(x=y_pred, y=sqrt_abs_residuals, ax=ax)
            ax.set_title(f"Scale-Location Plot - {label}")
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("sqrt(|Residuals|)")
        
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
    def pred_vs_true_plot(y_pred, y_true, y_pred_test=None, y_true_test=None, figsize=(10, 5)):
        if y_pred_test is not None and y_true_test is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            ax2.scatter(y_true, y_pred, alpha=0.6)
            m, M = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
            ax2.plot([m, M], [m, M], 'r--')
            ax2.set_title('Train Data')
            ax2.set_xlabel('True Values')
            ax2.set_ylabel('Predicted Values')
            
            ax1.scatter(y_true_test, y_pred_test, alpha=0.6)
            m, M = min(np.min(y_true_test), np.min(y_pred_test)), max(np.max(y_true_test), np.max(y_pred_test))
            ax1.plot([m, M], [m, M], 'r--')
            ax1.set_title('Test Data')
            ax1.set_xlabel('True Values')
            ax1.set_ylabel('Predicted Values')
            
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=figsize)
            plt.scatter(y_true, y_pred, alpha=0.6)
            m, M = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
            plt.plot([m, M], [m, M], 'r--')
            plt.title('Train Data')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.tight_layout()
            plt.show()
        plt.close()
        