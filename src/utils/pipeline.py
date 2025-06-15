# src/utils/pipeline.py

import pandas as pd
from features.composite  import add_selected_features
from features.registry   import FEATURE_FUNCTIONS, SESSION_FUNCTIONS

BASE_COLS = {'open','high','low','close','volume'}

def apply_feature_engineering(
    resampled: dict[str, pd.DataFrame],
    timeframes: list[str] = None,
    base_tf:     str      = None,
    features:    list[str]= None
) -> pd.DataFrame:
    """
    Given a dict of raw OHLCV DataFrames (keyed by timeframe),
    runs your selected features on each one, suffixes them by tf,
    applies session funcs on the base_tf, and then asof-merges all.
    
    Args:
      - resampled: e.g. {'5min': df5, '15min': df15, '1h': df1h}
      - timeframes: list of keys to process (defaults to list(resampled.keys()))
      - base_tf:    which tf to use as the merge base (defaults to first)
      - features:   which feature‐keys from FEATURE_FUNCTIONS to apply
                    (defaults to all keys in FEATURE_FUNCTIONS)
    """
    timeframes = timeframes or list(resampled.keys())
    base_tf     = base_tf     or timeframes[0]
    features    = features    or list(FEATURE_FUNCTIONS.keys())

    # 1) generate features per timeframe
    feature_dfs = {}
    for tf in timeframes:
        df0 = resampled[tf].copy()
        # inject only the features the user asked for
        df1 = add_selected_features(df0, features=features)
        # compute which cols are new (everything except base OHLCV)
        new_cols = [c for c in df1.columns if c not in BASE_COLS]
        # suffix them so they won't collide across timeframes
        suffix = f"_{tf}"
        df1 = df1.rename(columns={c: c + suffix for c in new_cols})
        feature_dfs[tf] = df1

    # 2) apply session/time features on the base tf
    df_base = feature_dfs[base_tf].copy()
    for fn in SESSION_FUNCTIONS.values():
        df_base = fn(df_base)

    # 3) asof-merge all the other‐TF suffix columns back into base
    df_merged = df_base.sort_index()
    for tf in timeframes:
        if tf == base_tf:
            continue
        suffix = f"_{tf}"
        cols_to_merge = [c for c in feature_dfs[tf].columns if c.endswith(suffix)]
        df_other = feature_dfs[tf][cols_to_merge].sort_index()
        df_merged = pd.merge_asof(
            df_merged, df_other,
            left_index=True, right_index=True,
            direction='backward'
        )

    return df_merged
