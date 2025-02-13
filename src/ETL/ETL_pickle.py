import os
import joblib


def ETL_pickle(config, model_config, n_rows=None, save_processed_data=True):
    """
    Extract, transform, and load (ETL) the data.

    Args:
        - config (dict): The configuration dictionary containing the ETL paths.

    Returns:
        - data (pandas DataFrame): The loaded and processed data.
    """
    raw_data_path = config["etl"]["raw_data_path"]
    processed_data_path = config["etl"]["processed_data_path"]
    data_batch_size = config["etl"]["data_batch_size"]
    data = joblib.load(raw_data_path)
    if n_rows is not None and isinstance(n_rows, int):
        data = data.head(n_rows)
    else:
        if data_batch_size == "all":
            if save_processed_data:
                os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
                joblib.dump(data, processed_data_path)
        else:
            if isinstance(data_batch_size, int):
                data = data.head(
                    data_batch_size
                )  # Example: limit to 1000 rows for testing
                if save_processed_data:
                    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
                    joblib.dump(data, processed_data_path)
                    print(f"Raw data saved to {processed_data_path}.")
            else:
                print("Data batch size is not specified or is of an invalid type.")

    return data
