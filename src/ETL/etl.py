import pandas as pd
import os
import joblib


def load_data(input_path, output_path):
    # Simulate loading data
    data = joblib.load(input_path)
    # Save processed data
    joblib.dump(data, output_path, index=False)
    print(f"Data saved to {output_path}")


import pandas as pd
import json
import pickle
import boto3
import h5py
import os
import requests
import sqlite3
from sqlalchemy import create_engine

# Function to load data from a CSV file
def load_csv(file_path: str):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Data loaded from the CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


# Function to load data from a Pickle file
def load_pickle(file_path: str):
    """
    Loads data from a Pickle file.

    Args:
    - file_path (str): Path to the Pickle file.

    Returns:
    - object: Python object (e.g., DataFrame, model) stored in the Pickle file.
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"Error loading Pickle file: {e}")
        return None


# Function to load data from a SQL database
def load_sql(query: str, db_url: str):
    """
    Loads data from a SQL database into a pandas DataFrame.

    Args:
    - query (str): SQL query to retrieve the data.
    - db_url (str): Database URL (e.g., 'postgresql://user:password@localhost/dbname').

    Returns:
    - pd.DataFrame: Data loaded from the SQL database.
    """
    try:
        engine = create_engine(db_url)
        data = pd.read_sql(query, engine)
        return data
    except Exception as e:
        print(f"Error loading data from SQL: {e}")
        return None


# Function to load data from a MongoDB database
def load_mongo(
    collection_name: str, db_name: str, host: str = "localhost", port: int = 27017
):
    """
    Loads data from a MongoDB collection.

    Args:
    - collection_name (str): Name of the collection to fetch data from.
    - db_name (str): MongoDB database name.
    - host (str, optional): Host for the MongoDB server (default is 'localhost').
    - port (int, optional): Port for the MongoDB server (default is 27017).

    Returns:
    - list: List of documents in the collection.
    """
    try:
        from pymongo import MongoClient

        client = MongoClient(host, port)
        db = client[db_name]
        collection = db[collection_name]
        data = list(collection.find())
        return data
    except Exception as e:
        print(f"Error loading data from MongoDB: {e}")
        return None


# Function to load data from an API (RESTful)
def load_api(url: str):
    """
    Loads data from a given API URL.

    Args:
    - url (str): URL of the API.

    Returns:
    - dict: Data returned by the API (usually in JSON format).
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error loading data from API: {e}")
        return None


# Function to load data from AWS S3
def load_s3(bucket_name: str, file_key: str, local_path: str):
    """
    Downloads a file from AWS S3 and saves it locally.

    Args:
    - bucket_name (str): Name of the S3 bucket.
    - file_key (str): S3 file key (path to the file within the bucket).
    - local_path (str): Local path where the file will be saved.

    Returns:
    - bool: True if file is downloaded successfully, False otherwise.
    """
    try:
        s3 = boto3.client("s3")
        s3.download_file(bucket_name, file_key, local_path)
        print(f"File downloaded to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return False


# Function to load data from HDF5 file
def load_hdf5(file_path: str, dataset_name: str):
    """
    Loads data from an HDF5 file.

    Args:
    - file_path (str): Path to the HDF5 file.
    - dataset_name (str): Name of the dataset within the HDF5 file.

    Returns:
    - numpy.ndarray: Data loaded from the HDF5 file.
    """
    try:
        with h5py.File(file_path, "r") as f:
            data = f[dataset_name][:]
        return data
    except Exception as e:
        print(f"Error loading data from HDF5 file: {e}")
        return None


# Function to load data from a Parquet file
def load_parquet(file_path: str):
    """
    Loads data from a Parquet file into a pandas DataFrame.

    Args:
    - file_path (str): Path to the Parquet file.

    Returns:
    - pd.DataFrame: Data loaded from the Parquet file.
    """
    try:
        data = pd.read_parquet(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from Parquet file: {e}")
        return None


# Function to load data from a JSON file
def load_json(file_path: str):
    """
    Loads data from a JSON file.

    Args:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Data loaded from the JSON file.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from JSON file: {e}")
        return None


# Function to load data from an Excel file
def load_excel(file_path: str):
    """
    Loads data from an Excel file into a pandas DataFrame.

    Args:
    - file_path (str): Path to the Excel file.

    Returns:
    - pd.DataFrame: Data loaded from the Excel file.
    """
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from Excel file: {e}")
        return None
