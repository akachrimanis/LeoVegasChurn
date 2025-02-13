import os
import shutil


def copy_file(source_file, destination_folder, overwrite=False):
    """
    Copies a specific file to a destination folder.

    Parameters:
        source_file (str): The full path of the file to be copied.
        destination_folder (str): The destination directory where the file will be copied.
        overwrite (bool): Whether to overwrite the file if it already exists in the destination. Default is False.

    Raises:
        ValueError: If the source file does not exist.
        IsADirectoryError: If the source file is a directory.
    """
    if not os.path.exists(source_file):
        raise ValueError(f"The source file '{source_file}' does not exist.")

    if os.path.isdir(source_file):
        raise IsADirectoryError(
            f"The source path '{source_file}' is a directory, not a file."
        )

    # Ensure the destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    # Construct the destination file path
    destination_file = os.path.join(destination_folder, os.path.basename(source_file))

    if not overwrite and os.path.exists(destination_file):
        print(f"File '{destination_file}' already exists. Skipping...")
    else:
        shutil.copy2(source_file, destination_file)
        print(f"Copied file '{source_file}' to '{destination_file}'.")
