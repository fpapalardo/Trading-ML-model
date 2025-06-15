# features/composite.py
from .registry import FEATURE_FUNCTIONS

def add_selected_features(df, features):
    """
    features can be:
      - None            → apply all
      - ["lagged", …]   → each name must be in FEATURE_FUNCTIONS
      - [("lagged", {"cols": […], "lags": (…) }), …]
      - [callable, …]
    """
    if features is None:
        features = list(FEATURE_FUNCTIONS.keys())

    for feat in features:
        if isinstance(feat, str):
            func = FEATURE_FUNCTIONS.get(feat)
            if func is None:
                raise ValueError(f"Unknown feature '{feat}'")
            df = func(df)

        elif isinstance(feat, tuple) and len(feat) == 2:
            name, kwargs = feat
            func = FEATURE_FUNCTIONS.get(name)
            if func is None:
                raise ValueError(f"Unknown feature '{name}'")
            df = func(df, **kwargs)

        elif callable(feat):
            df = feat(df)

        else:
            raise ValueError(f"Invalid feature spec: {feat!r}")

    return df
