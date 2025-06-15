import os
import platform
import pandas as pd
import matplotlib.pyplot as plt
from src.features.registry import FEATURE_FUNCTIONS, SESSION_FUNCS
from src.features.composite  import add_selected_features

def load_and_resample_data(
    market_hint,
    timeframes=['5min', '15min', '1h'],
    root_folder="./../data/"
):
    # Set emoji-compatible font based on OS
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Segoe UI Emoji'
    elif system == 'Linux':
        plt.rcParams['font.family'] = 'Noto Color Emoji'

    # Locate the folder for this market
    target_folder = None
    for sub in os.listdir(root_folder):
        full = os.path.join(root_folder, sub)
        if os.path.isdir(full) and market_hint.lower() in sub.lower():
            target_folder = full
            break
    if target_folder is None:
        raise FileNotFoundError(f"No folder in {root_folder} matches '{market_hint}'")
    print(f"Found folder: {target_folder}")

    # Load CSV files (comma-separated), select only timestamp and OHLCV
    df_list = []
    for fn in os.listdir(target_folder):
        if not fn.lower().endswith('.csv'):
            continue
        path = os.path.join(target_folder, fn)
        df_temp = pd.read_csv(
            path,
            usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['datetime'],
            sep=','
        )
        df_temp.rename(columns={'datetime': 'datetime'}, inplace=True)
        df_list.append(df_temp)
    if not df_list:
        raise ValueError("No CSV data files found in target folder.")

    # Concatenate and normalize
    df = pd.concat(df_list, ignore_index=True)
    df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
    df.drop_duplicates(subset='datetime', keep='first', inplace=True)
    df.sort_values('datetime', inplace=True)

    # Round OHLC to two decimals, cast volume
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float).round(2)
    df['volume'] = df['volume'].astype(float)

    # Set datetime index
    df.set_index('datetime', inplace=True)

    # Resample to desired timeframes
    valid = {
        '1min':'1min','3min':'3min','5min':'5min','15min':'15min',
        '30min':'30min','1h':'1h','2h':'2h','4h':'4h','6h':'6h','1d':'1d'
    }
    resampled = {}
    for tf in timeframes:
        rule = valid.get(tf.lower())
        if rule is None:
            raise ValueError(f"Unsupported timeframe: {tf}")
        df.index = df.index + pd.Timedelta(minutes=1)
        resampled_df = df.resample(rule, label='right', closed='right').agg({
            'open':'first',
            'high':'max',
            'low':'min',
            'close':'last',
            'volume':'sum'
        }).dropna()
        resampled_df.index = resampled_df.index - pd.Timedelta(minutes=5)
        resampled[tf] = resampled_df

    return df, resampled

def apply_feature_engineering(
    resampled: dict,
    timeframes: list      = None,
    base_tf:     str       = None,
    features:    list      = None
) -> pd.DataFrame:
    """
    - resampled: dict[str→DataFrame] of raw bars per timeframe
    - features:  list of keys from FEATURE_FUNCTIONS dict to apply
    - base_tf:   which tf to use as the merge base
    """
    timeframes  = timeframes or list(resampled.keys())
    base_tf      = base_tf     or timeframes[0]
    features     = features    or list(FEATURE_FUNCTIONS.keys())

    # 1) for each timeframe, call composite.add_selected_features
    feature_dfs = {}
    for tf in timeframes:
        df = resampled[tf].copy()
        suffix = f"_{tf}"
        # composite will look up each key in FEATURE_FUNCTIONS
        df = add_selected_features(df, features=features, suffix=suffix)
        feature_dfs[tf] = df

    # 2) session/time features on base tf
    df_base = feature_dfs[base_tf].copy()
    for fn in SESSION_FUNCTIONS.values():
        df_base = fn(df_base)

    # 3) merge all other‐tf suffix columns back into base
    df_merged = df_base.sort_index()
    for tf in timeframes:
        if tf == base_tf:
            continue
        suffix = f"_{tf}"
        cols   = [c for c in feature_dfs[tf].columns if c.endswith(suffix)]
        df_to  = feature_dfs[tf][cols].sort_index()
        df_merged = pd.merge_asof(
            df_merged, df_to,
            left_index=True, right_index=True,
            direction='backward'
        )

    return df_merged
