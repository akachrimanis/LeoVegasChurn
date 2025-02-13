import pandas as pd


def read_json_data(
    file_path: str,
    orient: str = "records",
    dtype: dict = None,
    convert_dates: bool = True,
    encoding: str = "utf-8",
    lines: bool = False,
    chunksize: int = None,
) -> pd.DataFrame:
    """
    Read data from a JSON file and return as a DataFrame with additional parameters for flexibility.

    Args:
        file_path (str): Path to the JSON file.
        orient (str, optional): Format of the JSON file. Options are 'split', 'records', 'index', 'columns', 'values'. Default is 'records'.
        dtype (dict, optional): Data types to assign to columns. Default is None.
        convert_dates (bool, optional): Whether to convert date columns to datetime objects. Default is True.
        encoding (str, optional): Encoding of the file. Default is 'utf-8'.
        lines (bool, optional): If True, the file is expected to contain JSON objects per line. Default is False.
        chunksize (int, optional): Number of rows to read at a time. If None, the entire file is read. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """
    try:
        data = pd.read_json(
            file_path,
            orient=orient,
            dtype=dtype,
            convert_dates=convert_dates,
            encoding=encoding,
            lines=lines,
            chunksize=chunksize,
        )
        if chunksize:
            print(f"Reading {chunksize} rows at a time from {file_path}.")
        else:
            print(f"Data extracted successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        raise
