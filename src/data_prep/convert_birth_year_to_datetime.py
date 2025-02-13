import pandas as pd
import numpy as np
from datetime import datetime
import math
def convert_birth_year_to_datetime(df, column_name='birth_year'):
    """
    Converts a column representing birth years (as integers) to datetime.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing birth years.

    Returns:
        pd.DataFrame: DataFrame with the birth year column converted to datetime.
    """
    if column_name in df.columns:
        try:
            # Convert the integer year to a datetime, defaulting to January 1st of that year
            df[column_name] = pd.to_datetime(df[column_name], format='%Y', errors='coerce')
            print(f"Column '{column_name}' successfully converted to datetime.")
        except Exception as e:
            print(f"Error converting column '{column_name}': {e}")
    else:
        print(f"Column '{column_name}' not found in DataFrame.")
    
    return df


def calculate_age(df, birth_date_column='birth_year', age_column='age'):
    """
    Calculates age from a datetime birth year column and adds it to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        birth_date_column (str): The name of the column with birth dates (in datetime format).
        age_column (str): The name of the column to store the calculated age.

    Returns:
        pd.DataFrame: DataFrame with an additional 'age' column.
    """
    if birth_date_column in df.columns:
        try:
            # Get the current date
            today = pd.Timestamp(datetime.today().date())
            
            # Calculate the age
            df[age_column] = df[birth_date_column].apply(
                lambda birth_date: math.ceil(
                    (today.year - birth_date.year) - 
                    ((today.month, today.day) < (birth_date.month, birth_date.day))
                )
            )
            
            print(f"Age successfully calculated and stored in '{age_column}'.")
        except Exception as e:
            print(f"Error calculating age from column '{birth_date_column}': {e}")
    else:
        print(f"Column '{birth_date_column}' not found in DataFrame.")
    
    return df
