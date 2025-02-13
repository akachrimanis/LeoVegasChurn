import pandas as pd
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def detect_outliers_iqr(df, columns):
    """
    Detects outliers in specified columns of a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of columns to check for outliers.

    Returns:
        dict: A dictionary with column names as keys and lists of outlier indices as values.
    """
    outliers = {}

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            outliers[col] = outlier_indices

            print(f"Outliers detected in '{col}': {len(outlier_indices)}")

    return outliers


def detect_outliers_zscore(df, columns, threshold=3):
    """
    Detects outliers in specified columns of a DataFrame using the Z-score method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of columns to check for outliers.
        threshold (float): Z-score threshold to identify outliers (default = 3).

    Returns:
        dict: A dictionary with column names as keys and lists of outlier indices as values.
    """
    outliers = {}

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_indices = df[(z_scores > threshold)].index.tolist()
            outliers[col] = outlier_indices

            print(f"Outliers detected in '{col}': {len(outlier_indices)}")

    return outliers





# Function to detect outliers using IQR and plot
def detect_outliers_iqr_with_plots(df, columns):
    """
    Detects outliers in specified columns of a DataFrame using the IQR method and plots them in red.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of columns to check for outliers.

    Returns:
        dict: A dictionary with column names as keys and lists of outlier indices as values.
    """
    outliers = {}

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            outliers[col] = outlier_indices

            print(f"Outliers detected in '{col}': {len(outlier_indices)}")

            # Plotting with outliers highlighted in red
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col], color='skyblue', width=0.5)
            
            # Highlight outliers in red
            outlier_values = df.loc[outlier_indices, col]
            plt.scatter(outlier_values, [0] * len(outlier_values), color='red', s=100, label='Outlier')

            plt.title(f'{col} with Outliers Highlighted (IQR)')
            plt.xlabel(col)
            plt.legend(['Outlier'], loc='upper right')
            plt.show()

    return outliers


# Function to detect outliers using Z-score and plot
def detect_outliers_zscore_with_plots(df, columns, threshold=3):
    """
    Detects outliers in specified columns of a DataFrame using the Z-score method and plots them in red.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of columns to check for outliers.
        threshold (float): Z-score threshold to identify outliers (default = 3).

    Returns:
        dict: A dictionary with column names as keys and lists of outlier indices as values.
    """
    outliers = {}

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_indices = df[z_scores > threshold].index.tolist()
            outliers[col] = outlier_indices

            print(f"Outliers detected in '{col}': {len(outlier_indices)}")

            # Plotting with outliers highlighted in red
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col], color='lightgreen', width=0.5)

            # Highlight outliers in red
            outlier_values = df.loc[outlier_indices, col]
            plt.scatter(outlier_values, [0] * len(outlier_values), color='red', s=100, label='Outlier')

            plt.title(f'{col} with Outliers Highlighted (Z-Score)')
            plt.xlabel(col)
            plt.legend(['Outlier'], loc='upper right')
            plt.show()

    return outliers
