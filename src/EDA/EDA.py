import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(data, target_column=None):
    """
    Perform Exploratory Data Analysis (EDA) on a dataset.

    Args:
        data (pd.DataFrame): The dataset to analyze.
        target_column (str, optional): The name of the target column, if applicable.
    """
    try:
        print("Basic Information")
        print("-" * 50)
        print(f"Shape of the dataset: {data.shape}")
        print("\nDataset Info:")
        print(data.info())
        print("\nFirst 5 rows:")
        print(data.head())

        print("\nSummary Statistics:")
        print(data.describe(include="all"))

        # Check for missing values
        print("\nMissing Values:")
        print(data.isnull().sum())

        # Check for duplicate rows
        print("\nDuplicate Rows:")
        print(data.duplicated().sum())

        # Data types
        print("\nData Types:")
        print(data.dtypes)

        # Distribution of numerical features
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        print("\nNumerical Features:", numerical_features)

        # Visualizing missing values
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()

        # Plot distributions of numerical columns
        for col in numerical_features:
            plt.figure(figsize=(8, 4))
            sns.histplot(data[col], kde=True, bins=30, color="blue")
            plt.title(f"Distribution of {col}")
            plt.show()

        # Boxplots for numerical features
        for col in numerical_features:
            plt.figure(figsize=(8, 4))
            sns.boxplot(data[col], color="orange")
            plt.title(f"Boxplot of {col}")
            plt.show()

        # Distribution of categorical features
        categorical_features = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        print("\nCategorical Features:", categorical_features)

        for col in categorical_features:
            plt.figure(figsize=(8, 4))
            sns.countplot(data[col], palette="Set2")
            plt.title(f"Countplot of {col}")
            plt.xticks(rotation=45)
            plt.show()

        # Correlation heatmap
        if numerical_features:
            plt.figure(figsize=(10, 8))
            corr = data[numerical_features].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Correlation Heatmap")
            plt.show()

        # Pairplot
        if numerical_features and len(numerical_features) <= 5:
            sns.pairplot(data[numerical_features])
            plt.title("Pairplot of Numerical Features")
            plt.show()

        # Relationship with the target column
        if target_column:
            print("\nAnalyzing the target column:", target_column)

            # Target column distribution
            if data[target_column].dtype in ["int64", "float64"]:
                plt.figure(figsize=(8, 4))
                sns.histplot(data[target_column], kde=True, color="green")
                plt.title(f"Distribution of {target_column}")
                plt.show()
            else:
                plt.figure(figsize=(8, 4))
                sns.countplot(data[target_column], palette="Set3")
                plt.title(f"Countplot of {target_column}")
                plt.xticks(rotation=45)
                plt.show()

            # Correlation with numerical features
            if numerical_features:
                plt.figure(figsize=(10, 4))
                corr_with_target = data[numerical_features].corrwith(
                    data[target_column]
                )
                corr_with_target.sort_values(ascending=False).plot(
                    kind="bar", color="coral"
                )
                plt.title(f"Correlation with Target: {target_column}")
                plt.show()

    except Exception as e:
        print(f"An error occurred during EDA: {e}")
