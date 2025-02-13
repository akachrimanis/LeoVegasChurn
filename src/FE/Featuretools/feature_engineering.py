import featuretools as ft
import pandas as pd
import numpy as np

# 1. Creating an EntitySet
def create_entityset(dataframe, entity_name, index_col):
    """
    Create an EntitySet from a DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        entity_name (str): The name of the entity (table).
        index_col (str): The column to use as the index.

    Returns:
        es (EntitySet): The created entity set.
    """
    try:
        es = ft.EntitySet(id="data_entityset")
        es = es.entity_from_dataframe(
            entity_name=entity_name, dataframe=dataframe, index=index_col
        )
        return es
    except Exception as e:
        print(f"Error in create_entityset: {e}")
        return None


# 2. Feature Engineering using Deep Feature Synthesis
def deep_feature_synthesis(
    es, target_entity, agg_primitives=None, trans_primitives=None
):
    """
    Perform Deep Feature Synthesis (DFS) to generate features for the target entity.

    Args:
        es (EntitySet): The input entity set.
        target_entity (str): The entity name for which we want to generate features.
        agg_primitives (list, optional): List of aggregation primitives to use. Defaults to None.
        trans_primitives (list, optional): List of transformation primitives to use. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with generated features.
    """
    try:
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_entity=target_entity,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
        )
        return feature_matrix
    except Exception as e:
        print(f"Error in deep_feature_synthesis: {e}")
        return None


# 3. Apply Transformation Primitives
def apply_transformation_primitive(es, target_entity, primitive):
    """
    Apply a specific transformation primitive on the target entity.

    Args:
        es (EntitySet): The input entity set.
        target_entity (str): The entity name for which to apply the transformation.
        primitive (str): The transformation primitive to apply (e.g., 'add_numeric' for adding numbers).

    Returns:
        pd.DataFrame: DataFrame with the transformed features.
    """
    try:
        feature_matrix, feature_defs = ft.dfs(
            entityset=es, target_entity=target_entity, trans_primitives=[primitive]
        )
        return feature_matrix
    except Exception as e:
        print(f"Error in apply_transformation_primitive: {e}")
        return None


# 4. Apply Aggregation Primitives
def apply_aggregation_primitive(es, target_entity, primitive):
    """
    Apply a specific aggregation primitive on the target entity.

    Args:
        es (EntitySet): The input entity set.
        target_entity (str): The entity name for which to apply the aggregation.
        primitive (str): The aggregation primitive to apply (e.g., 'mean' or 'sum').

    Returns:
        pd.DataFrame: DataFrame with the aggregated features.
    """
    try:
        feature_matrix, feature_defs = ft.dfs(
            entityset=es, target_entity=target_entity, agg_primitives=[primitive]
        )
        return feature_matrix
    except Exception as e:
        print(f"Error in apply_aggregation_primitive: {e}")
        return None


# 5. Custom Feature Generation (using Primitives)
def generate_custom_features(
    es,
    target_entity,
    agg_primitives=None,
    trans_primitives=None,
    custom_primitives=None,
):
    """
    Generate custom features for the target entity using a combination of aggregation and transformation primitives.

    Args:
        es (EntitySet): The input entity set.
        target_entity (str): The entity name for which to generate features.
        agg_primitives (list, optional): List of aggregation primitives to apply. Defaults to None.
        trans_primitives (list, optional): List of transformation primitives to apply. Defaults to None.
        custom_primitives (list, optional): List of custom primitives to apply. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with custom features.
    """
    try:
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_entity=target_entity,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            custom_primitives=custom_primitives,
        )
        return feature_matrix
    except Exception as e:
        print(f"Error in generate_custom_features: {e}")
        return None


# 6. Handling Time-Based Features
def generate_time_features(es, target_entity, time_column):
    """
    Generate time-based features for the target entity (e.g., month, day, etc.).

    Args:
        es (EntitySet): The input entity set.
        target_entity (str): The entity name for which to generate time-based features.
        time_column (str): The column containing datetime values.

    Returns:
        pd.DataFrame: DataFrame with time-based features.
    """
    try:
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_entity=target_entity,
            trans_primitives=["month", "day", "weekday", "hour"],
            where_columns=[time_column],
        )
        return feature_matrix
    except Exception as e:
        print(f"Error in generate_time_features: {e}")
        return None


# 7. Apply Custom Function
def apply_custom_function_to_entity(es, target_entity, custom_function):
    """
    Apply a custom user-defined function to generate features.

    Args:
        es (EntitySet): The input entity set.
        target_entity (str): The entity name to apply the function to.
        custom_function (function): A custom function that takes a DataFrame and returns a DataFrame.

    Returns:
        pd.DataFrame: DataFrame with custom features.
    """
    try:
        # Custom function applies to the entity
        df = es[target_entity].df
        transformed_df = custom_function(df)
        return transformed_df
    except Exception as e:
        print(f"Error in apply_custom_function_to_entity: {e}")
        return None
