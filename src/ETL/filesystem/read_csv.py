import pandas as pd
from typing import Optional


def read_csv(
    file_path: str,
    sep: str = ",",
    header: int = 0,
    index_col: Optional[str] = None,
    usecols: Optional[list[str]] = None,
    dtype: Optional[dict] = None,
    names: Optional[list] = None,
    skiprows: Optional[int] = None,
    nrows: Optional[int] = None,
    na_values: Optional[list] = None,
    parse_dates: Optional[list] = None,
    encoding: str = "utf-8",
    date_parser: callable = None,
    low_memory: bool = True,
    skip_blank_lines: bool = True,
    thousands: Optional[str] = None,
    comment: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read data from a CSV file and return as a DataFrame, with flexible parameters for handling various types of files.

    Args:
        file_path (str): Path to the CSV file.
        sep (str, optional): The delimiter separating the values. Default is ','.
        header (int, optional): Row(s) to use as column names. Default is 0 (first row).
        index_col (str, optional): Column(s) to use as the index. Default is None.
        usecols (list, optional): Columns to read from the file. Default is None (reads all columns).
        dtype (dict, optional): Data types to assign to columns. Default is None.
        names (list, optional): List of column names to use if the file does not have a header. Default is None.
        skiprows (int, optional): Number of rows to skip at the start. Default is None.
        nrows (int, optional): Number of rows to read. Default is None.
        na_values (list, optional): List of values to interpret as NaN. Default is None.
        parse_dates (list, optional): Columns to parse as dates. Default is None.
        encoding (str, optional): File encoding. Default is 'utf-8'.
        date_parser (callable, optional): Custom date parser function. Default is None.
        low_memory (bool, optional): Whether to process the file in chunks. Default is True.
        skip_blank_lines (bool, optional): Whether to skip blank lines. Default is True.
        thousands (str, optional): Character to interpret as a thousands separator. Default is None.
        comment (str, optional): Character to indicate comments in the file. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """
    try:
        # Read the CSV with the specified parameters
        data = pd.read_csv(
            file_path,
            sep=sep,
            header=header,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            names=names,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            parse_dates=parse_dates,
            encoding=encoding,
            date_parser=date_parser,
            low_memory=low_memory,
            skip_blank_lines=skip_blank_lines,
            thousands=thousands,
            comment=comment,
        )
        print(f"Data extracted successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        raise
