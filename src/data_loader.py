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
