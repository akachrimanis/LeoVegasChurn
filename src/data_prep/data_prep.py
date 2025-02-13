from src.data_prep.change_variable_format import convert_integers_to_float
from src.data_prep.missing_values import impute_selected_columns


def data_prep(data, config, model_config):
    """
    Prepare the data by cleaning and filling missing values.

    Args:
        - data (pandas DataFrame): The raw data to prepare.
        - config (dict): The configuration dictionary.

    Returns:
        - data (pandas DataFrame): The prepared data.
    """
    print("Preparing data...")

    # Drop duplicate rows

    data = data.drop_duplicates()
    data = data.drop(columns=config["data_prep"]["drop_cols"], axis=1)

    # Fill missing values with mean (for simplicity in this example)
    # data.fillna(data.mean(), inplace=True)
    data = convert_integers_to_float(data)
    data = impute_selected_columns(data, strategy="mean")
    # categorical_columns, target_column, categories = prepare_column_lists(df, target_column='target', ordinal_columns=ordinal_columns)
    return data
