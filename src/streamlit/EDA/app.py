import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.load_config import load_config
from src.ETL.ETL_pickle import ETL_pickle  # Importing ETL_pickle function
import os
import warnings

# Suppress specific deprecation warnings related to st.pyplot
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*st.pyplot\(\) without providing a figure argument.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*st.pyplot\(\) without providing a figure argument.*",
)


def perform_eda(data, target_column=None):
    """
    Perform Exploratory Data Analysis (EDA) on a dataset using Streamlit.

    Args:
        data (pd.DataFrame): The dataset to analyze.
        target_column (str, optional): The name of the target column, if applicable.
    """
    try:
        # Display basic dataset information
        st.header("Basic Information")
        st.write(f"Shape of the dataset: {data.shape}")
        st.write("### Dataset Info:")
        st.write(data.info())
        st.write("### First 5 rows:")
        st.write(data.head())

        st.write("### Summary Statistics:")
        st.write(data.describe(include="all"))

        # Check for missing values
        st.write("### Missing Values:")
        st.write(data.isnull().sum())

        # Check for duplicate rows
        st.write("### Duplicate Rows:")
        st.write(data.duplicated().sum())

        # Data types
        st.write("### Data Types:")
        st.write(data.dtypes)

        # Distribution of numerical features
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        st.write(f"### Numerical Features: {numerical_features}")

        # Visualizing missing values
        st.write("### Missing Values Heatmap:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

        # Plot distributions of numerical columns
        for col in numerical_features:
            st.write(f"### Distribution of {col}:")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data[col], kde=True, bins=30, color="blue", ax=ax)
            st.pyplot(fig)

        # Boxplots for numerical features
        for col in numerical_features:
            st.write(f"### Boxplot of {col}:")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data[col], color="orange", ax=ax)
            st.pyplot(fig)

        # Distribution of categorical features
        categorical_features = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        st.write(f"### Categorical Features: {categorical_features}")

        for col in categorical_features:
            st.write(f"### Countplot of {col}:")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(
                data=data, x=col, palette="Set2", ax=ax, legend=False
            )  # Set legend=False to avoid warning
            plt.xticks(rotation=45)
            st.pyplot(fig)
        # Correlation heatmap
        if numerical_features:
            st.write("### Correlation Heatmap:")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = data[numerical_features].corr()
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax
            )
            st.pyplot(fig)

        # Pairplot (Handle with explicit figure)
        if numerical_features and len(numerical_features) <= 5:
            st.write("### Pairplot of Numerical Features:")
            # Pairplot does not need an explicit figure, but here we do it for consistency
            fig = sns.pairplot(data[numerical_features])
            st.pyplot(fig.fig)

        # Relationship with the target column
        if target_column:
            st.write(f"### Analyzing the target column: {target_column}")

            # Target column distribution
            if data[target_column].dtype in ["int64", "float64"]:
                st.write(f"### Distribution of {target_column}:")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(data[target_column], kde=True, color="green", ax=ax)
                st.pyplot(fig)
            else:
                st.write(f"### Countplot of {target_column}:")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(data[target_column], palette="Set3", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # Correlation with numerical features
            if numerical_features:
                st.write(f"### Correlation with Target: {target_column}")
                fig, ax = plt.subplots(figsize=(10, 4))
                corr_with_target = data[numerical_features].corrwith(
                    data[target_column]
                )
                corr_with_target.sort_values(ascending=False).plot(
                    kind="bar", color="coral", ax=ax
                )
                st.pyplot(fig)

    except Exception as e:
        st.write(f"An error occurred during EDA: {e}")


# Streamlit interface
def main():
    st.title("Exploratory Data Analysis (EDA) with Streamlit")
    config_file_path = "/Users/tkax/dev/aimonetize/WIP/ProjectTemplates/predictAPI/sales-forecasting/configs/config.yaml"
    config = load_config(config_file_path)
    model_type = config["info"]["model_type"]
    model_config_folder = config["info"]["model_config_folder"]
    model_config = load_config(os.path.join(model_config_folder, f"{model_type}.yaml"))

    if st.button("Load Data"):
        try:
            # Load data using the ETL_pickle function
            data = ETL_pickle(
                config, model_config, n_rows=None, save_processed_data=True
            )

            st.write("### Dataset Loaded:")
            st.write(data.head())

            # Target column selection
            target_column = st.selectbox(
                "Select Target Column (Optional)", options=[None] + list(data.columns)
            )

            # Perform EDA on the dataset
            perform_eda(data, target_column)

        except Exception as e:
            st.write(f"Error loading data: {e}")


if __name__ == "__main__":
    main()
