import os

def find_large_files(root_dir, size_threshold_mb=100):
    """
    Finds files larger than a given threshold while excluding hidden folders (.) and the 'data' folder.

    Parameters:
    - root_dir (str): The path to the root directory to scan.
    - size_threshold_mb (int): The minimum file size in MB to be considered large.

    Returns:
    - A list of large files with their sizes.
    """
    large_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude hidden folders and the 'data' folder
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != 'data']

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB
                if file_size_mb > size_threshold_mb:
                    large_files.append((file_path, file_size_mb))
            except Exception as e:
                print(f"Error accessing {file_path}: {e}")

    return large_files

# Example Usage
root_directory = "."  # Change this to the directory you want to scan
size_threshold = 10  # Files larger than 100MB

large_files = find_large_files(root_directory, size_threshold)

# Print Results
if large_files:
    print("\n🚀 Large Files Found:")
    for file_path, size in sorted(large_files, key=lambda x: x[1], reverse=True):
        print(f"{file_path} - {size:.2f} MB")
else:
    print("\n✅ No large files found.")
