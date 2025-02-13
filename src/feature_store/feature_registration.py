from datetime import timedelta
from feast import Entity, FeatureView, Field, ValueType
from feast.types import Int64, Float32, String, Bool
import pandas as pd
import os


def register_features_from_df(
    feature_view_name: str,
    df: pd.DataFrame,
    entity_name: str,
    entity_description: str,
    timestamp_field: str,
    feature_repo_path: str = ".",
    ttl_days: int = 1,
) -> None:
    """
    Registers features with Feast based on a Pandas DataFrame.

    Args:
        feature_view_name: The name of the feature view.
        df: The Pandas DataFrame containing the feature data.
        entity_name: The name of the entity column.
        entity_description: Description of the entity.
        timestamp_field: The name of the timestamp column.
        feature_repo_path: Path to the Feast repository.
        ttl_days: Time-to-live for the features in days.
    """
    try:
        from feast import FeatureStore
        from feast.infra.offline_stores.file_source import FileSource
    except ImportError:
        print("Feast is not installed. Install it with pip install feast")
        return None

    try:
        fs = FeatureStore(repo_path=feature_repo_path)
    except Exception as e:
        print(f"Error initializing feature store: {e}")
        return None

    if not os.path.exists(feature_repo_path):
        print(f"Feature repo not found at {feature_repo_path}")
        return None

    if not isinstance(df, pd.DataFrame):
        print("df must be a pandas DataFrame")
        return None

    if entity_name not in df.columns:
        print(f"Entity column '{entity_name}' not found in DataFrame")
        return None

    if timestamp_field not in df.columns:
        print(f"Timestamp column '{timestamp_field}' not found in DataFrame")
        return None

    entity = Entity(
        name=entity_name, value_type=ValueType.INT64, description=entity_description
    )

    features = []
    for column_name, dtype in df.dtypes.items():
        if column_name in [entity_name, timestamp_field]:
            continue  # Skip entity and timestamp columns

            if pd.api.types.is_integer_dtype(dtype):
                feast_type = Int64
            else:
                feast_type = Float32
        elif pd.api.types.is_string_dtype(dtype):
            feast_type = String
        elif pd.api.types.is_bool_dtype(dtype):
            feast_type = Bool
        else:
            feast_type = String  # Default to string

        features.append(Field(name=column_name, dtype=feast_type()))

    feature_view = FeatureView(
        name=feature_view_name,
        entities=[entity],
        ttl=timedelta(days=ttl_days),
        online=True,
        schema=features,
        source=FileSource(
            path=f"{feature_view_name}.parquet", timestamp_field=timestamp_field
        ),
    )

    try:
        fs.apply([feature_view])
        print(f"Feature view '{feature_view_name}' registered successfully.")
    except Exception as e:
        print(f"Error registering feature view: {e}")
