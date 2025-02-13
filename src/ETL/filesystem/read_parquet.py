import pandas as pd


def read_parquet_data(
    file_path: str,
    columns: list = None,
    engine: str = "auto",
    use_threads: bool = True,
    filters: list = None,
    categories: list = None,
    index_col: str = None,
) -> pd.DataFrame:
    """
    Read data from a Parquet file and return as a DataFrame with additional parameters for flexibility.

    Args:
        file_path (str): Path to the Parquet file.
        columns (list, optional): List of columns to select from the Parquet file. If None, all columns are read.
        engine (str, optional): The engine to use for reading the Parquet file. Options are 'auto', 'pyarrow', 'fastparquet'. Default is 'auto'.
        use_threads (bool, optional): Whether to use multi-threading for reading the file. Default is True.
        filters (list, optional): List of filters to apply on the data. Default is None.
        categories (list, optional): List of columns to cast to categorical data. Default is None.
        index_col (str, optional): Column(s) to set as the index. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """
    try:
        data = pd.read_parquet(
            file_path,
            columns=columns,
            engine=engine,
            use_threads=use_threads,
            filters=filters,
            categories=categories,
            index_col=index_col,
        )
        print(f"Data extracted successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading Parquet file {file_path}: {e}")
        raise
