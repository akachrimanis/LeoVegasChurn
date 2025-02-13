import pandas as pd


def read_excel_data(
    file_path: str,
    sheet_name: str = None,
    header: int = 0,
    index_col: str = None,
    usecols: list = None,
    dtype: dict = None,
    names: list = None,
    skiprows: int = None,
    nrows: int = None,
    na_values: list = None,
    parse_dates: list = None,
    engine: str = "openpyxl",
    encoding: str = "utf-8",
    sheet_names: bool = False,
    skipfooter: int = 0,
    comment: str = None,
) -> pd.DataFrame:
    """
    Read data from an Excel file and return as a DataFrame, with flexible parameters for handling various formats.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str, optional): Name of the sheet to read. If None, reads the first sheet.
        header (int, optional): Row(s) to use as column names. Default is 0 (first row).
        index_col (str, optional): Column(s) to use as the index. Default is None.
        usecols (list, optional): Columns to read from the sheet. Default is None.
        dtype (dict, optional): Data types to assign to columns. Default is None.
        names (list, optional): List of column names to use if the sheet does not have a header. Default is None.
        skiprows (int, optional): Number of rows to skip at the start of the sheet. Default is None.
        nrows (int, optional): Number of rows to read. Default is None.
        na_values (list, optional): List of values to interpret as NaN. Default is None.
        parse_dates (list, optional): Columns to parse as dates. Default is None.
        engine (str, optional): Engine to use for reading. Default is 'openpyxl'.
        encoding (str, optional): File encoding. Default is 'utf-8'.
        sheet_names (bool, optional): Whether to return a list of all sheet names. Default is False.
        skipfooter (int, optional): Number of rows to skip at the end of the sheet. Default is 0.
        comment (str, optional): Character to indicate comments in the sheet. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the extracted data, or a list of sheet names if sheet_names=True.
    """
    try:
        if sheet_names:
            # Return the list of sheet names in the Excel file
            with pd.ExcelFile(file_path, engine=engine) as xl:
                sheet_list = xl.sheet_names
            print(f"Sheet names: {sheet_list}")
            return sheet_list

        # Read the Excel sheet with the specified parameters
        data = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            names=names,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            parse_dates=parse_dates,
            engine=engine,
            encoding=encoding,
            skipfooter=skipfooter,
            comment=comment,
        )
        print(f"Data extracted successfully from {file_path}, sheet: {sheet_name}")
        return data
    except Exception as e:
        print(f"Error reading Excel file {file_path}: {e}")
        raise
