import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(df, categorical_columns):
    """
    Transforms categorical variables into one-hot encoded columns.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - categorical_columns (list): List of column names to be one-hot encoded.

    Returns:
    - pd.DataFrame: Transformed dataframe with one-hot encoded columns.
    """
    encoder = OneHotEncoder(
        sparse_output=False, drop="first"
    )  # Avoiding multicollinearity with 'drop=first'
    encoded_features = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index,
    )
    # Concatenate the one-hot encoded columns back to the original dataframe
    df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    return df
