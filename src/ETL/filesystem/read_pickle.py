import pandas as pd
import pickle
from typing import Optional


def read_pickle_data(
    file_path: str,
    compression: Optional[str] = None,
    mode: str = "rb",
    errors: str = "strict",
) -> object:
    """
    Read data from a Pickle file and return the object stored in the pickle file.

    Args:
        file_path (str): Path to the Pickle file.
        encoding (str, optional): Encoding for text files, default is 'utf-8'. Used when loading a pickle with text data.
        compression (str, optional): Compression type if the pickle file is compressed (e.g., 'gzip', 'bz2', 'xz'). Default is None.
        mode (str, optional): Mode in which to open the file. Default is 'rb' (read binary).
        errors (str, optional): Error handling strategy. Default is 'strict'.

    Returns:
        object: The Python object stored in the pickle file (e.g., a DataFrame, model, etc.).
    """
    try:
        with open(file_path, mode) as f:
            if compression:
                # Handle compressed pickle files
                data = pd.read_pickle(f, compression=compression)
            else:
                data = pickle.load(f)  # Load the pickle data
            print(f"Pickle data successfully loaded from {file_path}")
            return data
    except Exception as e:
        print(f"Error reading Pickle file {file_path}: {e}")
        raise
