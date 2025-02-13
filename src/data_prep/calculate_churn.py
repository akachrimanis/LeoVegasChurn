import pandas as pd

def create_churn_column(df, end_date_col='end_date', max_date_col='max_date', churn_days=30):
    """
    Creates a 'churn' column based on the difference between end_date and max_date.

    Args:
        df: Pandas DataFrame containing the end_date and max_date columns.
        end_date_col: Name of the column containing the end date (datetime).
        max_date_col: Name of the column containing the maximum date (datetime).
        churn_days: Number of days of inactivity to define churn.

    Returns:
        Pandas DataFrame with the added 'churn' column.
        Returns the original DataFrame if there are any errors during processing.
    """

    try:
        # Ensure date columns are datetime objects
        df[end_date_col] = pd.to_datetime(df[end_date_col])
        df[max_date_col] = pd.to_datetime(df[max_date_col])

        # Calculate the difference in days
        df['date_difference'] = (df[max_date_col] - df[end_date_col]).dt.days

        # Create the churn column
        df['churn'] = (df['date_difference'] > churn_days).astype(int)

        # Drop the temporary date_difference column (optional, but good practice)
        df = df.drop('date_difference', axis=1)

        return df

    except (KeyError, TypeError) as e:  # Handle potential errors gracefully
        print(f"Error creating churn column: {e}")
        return df  # Return original DataFrame if error occurs