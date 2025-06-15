import os
import platform
import pandas as pd
import matplotlib.pyplot as plt

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
    add_all_features,
    add_time_session_features,
    #add_trend_features,
    timeframes: list = ['5min'],
    base_tf: str = '5min'
):
    print("--- Starting Feature Engineering Execution ---")

    # --- Input Validations ---
    if base_tf not in timeframes:
        raise ValueError(f"Base timeframe '{base_tf}' must be included in the `timeframes` list.")

    missing_tfs = [tf for tf in timeframes if tf not in resampled]
    if missing_tfs:
        raise ValueError(f"The following requested timeframes are missing from resampled data: {missing_tfs}")

    if not callable(add_all_features):
        raise NameError("Function 'add_all_features' must be provided.")
    if not callable(add_time_session_features):
        raise NameError("Function 'add_time_session_features' must be provided.")
    # if not callable(add_trend_features):
    #     raise NameError("Function 'add_trend_features' must be provided.")

    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    for tf in timeframes:
        df = resampled[tf]
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Timeframe '{tf}' missing required OHLCV columns.")

    # --- Generate Features ---
    feature_dfs = {}
    for tf in timeframes:
        suffix = f"_{tf}"
        print(f"\nGenerating features for timeframe: {tf} — shape: {resampled[tf].shape}")
        
        # First add standard features
        df_with_features = add_all_features(resampled[tf].copy(), suffix=suffix)
        
        # Then add trend features
        # print(f"Adding trend features for timeframe: {tf}")
        # df_with_features = add_trend_features(df_with_features, suffix=suffix)
        
        feature_dfs[tf] = df_with_features
        print(f"Output shape for {tf}: {feature_dfs[tf].shape}")

    # --- Add Time/Session Features to Base Timeframe ---
    print(f"\nAdding time/session features to base timeframe: {base_tf}")
    df_base = add_time_session_features(feature_dfs[base_tf].copy())

    # --- Merge All Other Timeframes ---
    print("\nMerging additional timeframe features into base...")
    df_merged = df_base.sort_index()

    for tf in timeframes:
        if tf == base_tf:
            continue

        suffix = f"_{tf}"
        cols_to_merge = [col for col in feature_dfs[tf].columns if col.endswith(suffix)]
        if not cols_to_merge:
            print(f"⚠️ No columns with suffix {suffix} in {tf}; skipping merge.")
            continue

        df_to_merge = feature_dfs[tf][cols_to_merge].sort_index()
        df_merged = pd.merge_asof(
            df_merged,
            df_to_merge,
            left_index=True,
            right_index=True,
            direction='backward'
        )

    print("--- Feature Engineering Execution COMPLETE ---")
    print(f"Final df_merged shape: {df_merged.shape}")
    return df_merged
