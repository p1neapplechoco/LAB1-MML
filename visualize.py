import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

class Visualizer:
    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, min_correlation: float, figsize=(10, 8), annot=True):
        numeric_data = df.select_dtypes(include=['number'])
        corr_matrix = numeric_data.corr()

        mask = abs(corr_matrix) < min_correlation
        filtered_corr = corr_matrix.mask(mask)

        plt.figure(figsize=figsize)
        sns.heatmap(filtered_corr, cmap="coolwarm", vmin=-1, vmax=1, annot=annot)
        plt.title(f"Feature Correlation Heatmap (|corr| >= {min_correlation})")
        plt.show()

    @staticmethod
    def target_feature_scatterplots(df: pd.DataFrame, target_col: str, cols_per_row=3, figsize=(5, 5)):
        numeric_data = df.select_dtypes(include=['number'])
        
        if target_col not in numeric_data:
            raise ValueError(f"'{target_col}' không phải là cột số hoặc không tồn tại trong DataFrame.")
        
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
