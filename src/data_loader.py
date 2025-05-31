import os
import platform
import pandas as pd
import matplotlib.pyplot as plt

def load_and_resample_data(folder_path="./../data/", timeframes=['5min', '15min', '1h']):
    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df_list = []

    # Set emoji-compatible font based on OS
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Segoe UI Emoji'
    elif system == 'Linux':
        plt.rcParams['font.family'] = 'Noto Color Emoji'

    # Read and concatenate files
    for filename in os.listdir(folder_path):
        if filename.endswith(('.csv', '.txt')):
            file_path = os.path.join(folder_path, filename)
            df_temp = pd.read_csv(file_path, sep=';', header=None, names=column_names, on_bad_lines='warn')
            df_list.append(df_temp)

    if not df_list:
        raise ValueError("No data files found in folder.")

    df = pd.concat(df_list, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/New_York')
    df = df.drop_duplicates(subset='datetime', keep='first').sort_values('datetime').reset_index(drop=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = df.set_index('datetime')

    # === Dynamic resampling with validated timeframes ===
    valid_timeframes = {
        '1min': '1min',
        '3min': '3min',
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '1d': '1d'
    }

    resampled = {}
    for tf in timeframes:
        rule = valid_timeframes.get(tf.lower())
        if not rule:
            raise ValueError(f"Unsupported timeframe: {tf}")

        resampled_df = df.resample(rule, label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        resampled[tf] = resampled_df


    return df, resampled

def apply_feature_engineering(
    resampled: dict,
    add_all_features,
    add_time_session_features,
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
        feature_dfs[tf] = add_all_features(resampled[tf].copy(), suffix=suffix)
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
