import inspect
import os
import pandas as pd
import importlib


def extract_function_info(base_path):
    """
    Extracts function names, docstrings, file paths, args, and kwargs from Python source files.

    Args:
        base_path: The root directory to search for Python files.

    Returns:
        A pandas DataFrame containing the extracted information, or None if an error occurs.
    """

    function_data = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    module_path = os.path.relpath(file_path, base_path).replace(
                        os.sep, "."
                    )[:-3]
                    spec = importlib.util.spec_from_file_location(
                        module_path, file_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj):
                            docstring = inspect.getdoc(obj) or ""
                            signature = inspect.signature(obj)
                            parameters = signature.parameters
                            args = []
                            kwargs = {}

                            for param_name, param in parameters.items():
                                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                                    args.append("*" + param_name)  # *args
                                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                                    kwargs[param_name] = "**"  # **kwargs
                                elif param.default is inspect.Parameter.empty:
                                    args.append(param_name)
                                else:
                                    kwargs[param_name] = param.default

                            function_data.append(
                                {
                                    "function_name": name,
                                    "docstring": docstring,
                                    "file_path": file_path,
                                    "module_path": module_path,
                                    "args": ", ".join(
                                        args
                                    ),  # Store args as a comma-separated string
                                    "kwargs": kwargs,  # Store kwargs as a dictionary
                                }
                            )
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    try:
        df = pd.DataFrame(function_data)
        return df
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None


if __name__ == "__main__":
    # Example usage:
    base_directory = "/Users/tkax/dev/aimonetize/WIP/AIAgents/src"  # Replace "." with the actual path to your source code.
    df_functions = extract_function_info(base_directory)

    if df_functions is not None:
        print(df_functions)
        df_functions.to_csv(
            os.path.join(base_directory, "data/function_info.csv"), index=False
        )
