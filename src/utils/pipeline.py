# src/utils/pipeline.py

import pandas as pd
from features.composite  import add_selected_features
from features.registry   import FEATURE_FUNCTIONS, SESSION_FUNCTIONS

BASE_COLS = {'open','high','low','close','volume'}

def apply_feature_engineering(
    resampled: dict[str, pd.DataFrame],
    timeframes: list[str] = None,
    base_tf: str = None,
    features: list[str] = None
) -> pd.DataFrame:
    """
    Backtesting version that matches live behavior exactly.
    """
    timeframes = timeframes or list(resampled.keys())
    base_tf = base_tf or timeframes[0]
    features = features or list(FEATURE_FUNCTIONS.keys())

    # 1) Generate features per timeframe
    feature_dfs = {}
    for tf in timeframes:
        df0 = resampled[tf].copy()
        df1 = add_selected_features(df0, features=features)
        new_cols = [c for c in df1.columns if c not in BASE_COLS]
        suffix = f"_{tf}"
        df1 = df1.rename(columns={c: c + suffix for c in new_cols})
        feature_dfs[tf] = df1

    # 2) Apply session features to base
    df_base = feature_dfs[base_tf].copy()
    for fn in SESSION_FUNCTIONS.values():
        df_base = fn(df_base)

    # 3) Merge with proper real-time alignment
    df_merged = df_base.sort_index()
    
    for tf in timeframes:
        if tf == base_tf:
            continue
            
        suffix = f"_{tf}"
        cols_to_merge = [c for c in feature_dfs[tf].columns if c.endswith(suffix)]
        df_other = feature_dfs[tf][cols_to_merge].sort_index()
        
        # CRITICAL: Use merge_asof with proper parameters to match live behavior
        # This ensures we only use candles that would be complete at each timestamp
        df_merged = pd.merge_asof(
            df_merged, 
            df_other,
            left_index=True, 
            right_index=True,
            direction='backward',
            # This is the key - we allow exact matches because a candle 
            # is available immediately after it closes
            allow_exact_matches=True
        )

    return df_merged

def apply_feature_engineering_live(
    df_ohlcv: pd.DataFrame,
    timeframes: list[str],
    base_tf: str,
    features: list[str] = None
) -> pd.DataFrame:
    """
    Live version - simplified to match backtest exactly.
    """
    # Step 1: Resample to create higher timeframes
    resampled = {}
    for tf in timeframes:
        if tf == base_tf:
            resampled[tf] = df_ohlcv.copy()
        else:
            # Use label='left' to match the default pandas behavior
            resampled[tf] = df_ohlcv.resample(tf).agg({
                'open': 'first', 
                'high': 'max', 
                'low': 'min', 
                'close': 'last', 
                'volume': 'sum'
            }).dropna()

    # Step 2-4: Apply the same logic as backtesting
    features = features or list(FEATURE_FUNCTIONS.keys())
    
    # Generate features
    feature_dfs = {}
    for tf in timeframes:
        df0 = resampled[tf].copy()
        df1 = add_selected_features(df0, features=features)
        new_cols = [c for c in df1.columns if c not in BASE_COLS]
        suffix = f"_{tf}"
        df1.rename(columns={c: c + suffix for c in new_cols}, inplace=True)
        feature_dfs[tf] = df1

    # Apply session functions
    df_base = feature_dfs[base_tf].copy()
    for fn in SESSION_FUNCTIONS.values():
        df_base = fn(df_base)

    # Merge using the same logic as backtest
    df_merged = df_base.sort_index()
    
    for tf in timeframes:
        if tf == base_tf:
            continue
            
        suffix = f"_{tf}"
        cols_to_merge = [c for c in feature_dfs[tf].columns if c.endswith(suffix)]
        df_other = feature_dfs[tf][cols_to_merge].sort_index()
        
        df_merged = pd.merge_asof(
            df_merged, 
            df_other,
            left_index=True, 
            right_index=True,
            direction='backward',
            allow_exact_matches=True
        )

    return df_merged