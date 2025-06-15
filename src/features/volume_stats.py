import pandas as pd
import numpy as np

# --- Extended Volume & CVD/Wick Stats ---

def add_volume_delta_rollsum(df, delta_col='volume', windows=(2,3)):
    """
    Append rolling sums of 1-bar delta over the last N bars for each N in windows.
    """
    df = df.copy()
    df['Vol_Delta_1'] = df[delta_col].diff(1)
    for n in windows:
        df[f'Vol_Delta_rollsum_{n}'] = df['Vol_Delta_1'].rolling(n).sum()
    return df


def add_cvd(df, delta_col='volume', lags=3):
    """
    Append Cumulative Volume Delta (CVD) over specified lag.
    """
    df = df.copy()
    signed = np.where(df['close'] >= df['open'], df[delta_col], -df[delta_col])
    df[f'CVD_{lags}'] = pd.Series(signed, index=df.index).rolling(lags).sum()
    return df


def add_wick_percent(df):
    """
    Append upper and lower wick as percent of candle range.
    """
    df = df.copy()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    upper = df['high'] - df[['open','close']].max(axis=1)
    lower = df[['open','close']].min(axis=1) - df['low']
    df['Upper_Wick_%'] = (upper / rng).fillna(0)
    df['Lower_Wick_%'] = (lower / rng).fillna(0)
    return df


def add_rel_volume(df, volume_col='volume', window=20):
    """
    Append relative volume versus rolling mean.
    """
    df = df.copy()
    mv = df[volume_col].rolling(window).mean()
    df['Rel_Vol_20'] = (df[volume_col] / mv).replace([np.inf, -np.inf], np.nan)
    return df