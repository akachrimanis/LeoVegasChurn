import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


def read_sql_data(
    connection_string: str,
    query: str,
    params: dict = None,
    chunksize: int = None,
    dialect: str = "mysql",
    pool_size: int = 5,
    max_overflow: int = 10,
) -> pd.DataFrame:
    """
    Read data from a SQL database using a connection string and query.

    Args:
        connection_string (str): SQLAlchemy connection string for the database.
        query (str): SQL query to fetch data.
        params (dict, optional): A dictionary of parameters to pass to the SQL query. Default is None.
        chunksize (int, optional): If specified, it will return an iterator and load data in chunks of the given size. Default is None.
        dialect (str, optional): The SQL dialect (e.g., 'mysql', 'postgresql', etc.). Default is 'mysql'.
        pool_size (int, optional): The size of the connection pool. Default is 5.
        max_overflow (int, optional): The maximum number of connections to allow in the pool. Default is 10.

    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """
    try:
        # Create an engine with the specified parameters
        engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
        )

        # If a chunksize is specified, the function will return an iterator for large queries
        if chunksize:
            data_iter = pd.read_sql(query, engine, params=params, chunksize=chunksize)
            data = pd.concat(data_iter)  # Combine all chunks into a single DataFrame
        else:
            data = pd.read_sql(query, engine, params=params)

        print(f"Data extracted successfully from SQL database using query: {query}")
        return data
    except SQLAlchemyError as e:
        print(f"Error reading data from SQL database: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
