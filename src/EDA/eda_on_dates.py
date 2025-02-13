import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_on_date_column(df, date_column):
    """
    Perform EDA on a date column: min, max, and frequency by month.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column (str): The name of the date column.

    Returns:
        None
    """
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Drop rows where date conversion failed
    df = df.dropna(subset=[date_column])

    # Get the minimum and maximum dates
    min_date = df[date_column].min()
    max_date = df[date_column].max()

    print(f"Minimum Date in '{date_column}': {min_date}")
    print(f"Maximum Date in '{date_column}': {max_date}")

    # Frequency of records per month
    monthly_freq = df[date_column].dt.to_period('M').value_counts().sort_index()

    print("\nFrequency of Records per Month:")
    print(monthly_freq)

    # Plotting frequency per month
    plt.figure(figsize=(12, 6))
    monthly_freq.plot(kind='bar', color='skyblue')
    plt.title(f'Frequency of Records per Month ({date_column})')
    plt.xlabel('Month')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
