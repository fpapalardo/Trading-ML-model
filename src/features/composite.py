# features/composite.py
from .registry import FEATURE_FUNCTIONS

def add_selected_features(df, features: list):
    """
    Add only selected features to df.
    Args:
        df: input DataFrame
        features: list of feature keys (str)
    Returns:
        df with selected features added
    """
    for feat in features:
        func = FEATURE_FUNCTIONS.get(feat)
        if func is None:
            raise ValueError(f"Unknown feature: {feat}")
        df = func(df)
    return df
