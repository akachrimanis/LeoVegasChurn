from src.FE.encoding import prepare_column_lists, encode_columns
from src.FE.date_features import create_extended_date_features
from src.data_prep.change_variable_format import convert_integers_to_float
from src.FE.normalize import scale_features
import joblib


def FE(data, config, date_columns, model_config):
    """
    Feature engineering step: encoding categorical variables and scaling.

    Args:
        - data (pandas DataFrame): The prepared data.
        - config (dict): The configuration dictionary containing variables to encode and scale.

    Returns:
        - data (pandas DataFrame): The engineered data with one-hot encoding, scaling, and integer conversion.
    """
    print("Performing feature engineering...")

    # One-hot encode categorical variables
    scaler_type = config["FE"]["encoding_params"]["scaler_type"]
    scaling_criteria = config["FE"]["encoding_params"]["scaling_criteria"]
    target_column = config["variables"]["y"]
    ordinal_columns = config["data_prep"]["ordinal_columns"]

    # Convert integers to floats where necessary
    data = convert_integers_to_float(data)
    data = create_extended_date_features(data, config["data_prep"]["date_columns"][0])
    categorical_columns, target_column, categories = prepare_column_lists(
        data, target_column=target_column, ordinal_columns=ordinal_columns
    )
    data = encode_columns(
        data, categorical_columns, target_column, categories, date_columns
    )

    # Scale features (standard scaling by default)
    try:
        data = scale_features(data, scaler_type, scaling_criteria)
        print("Scaled Data (Standard):\n", data.columns)
    except Exception as e:
        print(f"Failed to scale with StandardScaler: {e}")

    # Save the engineered data
    output_path = config["data_prep"]["engineered_data_path"]
    joblib.dump(data, output_path)
    print(f"Data prepared and saved to {output_path}.")

    return data
