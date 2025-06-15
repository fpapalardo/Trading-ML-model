import pandas as pd
import numpy as np
import pandas_ta as ta

# --- Momentum Indicators & Signals ---
def add_rsi_all(
    df: pd.DataFrame,
    lengths: tuple[int,int] = (7, 14)
) -> pd.DataFrame:
    """
    Compute RSI for each length in `lengths` (default 7 & 14)
    and append columns RSI_<length>.
    """
    for length in lengths:
        col = f"RSI_{length}"
        df.ta.rsi(length=length, append=True, col_names=(col,))
    return df

def add_rsi_signals_all(
    df: pd.DataFrame,
    lengths: tuple[int,int] = (7, 14),
    ob_level: int = 70,
    os_level: int = 30
) -> pd.DataFrame:
    """
    For each RSI_<length> in `lengths`, append
      RSI_<length>_OB (overbought), and
      RSI_<length>_OS (oversold).
    """
    for length in lengths:
        col = f"RSI_{length}"
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            df[f"{col}_OB"] = (s > ob_level).astype(int)
            df[f"{col}_OS"] = (s < os_level).astype(int)
    return df

def rsi_divergence_all(
    df: pd.DataFrame,
    lengths: tuple[int,int] = (7, 14),
    lookback: int = 5
) -> pd.DataFrame:
    """
    For each RSI_<length>, append RSI_<length>_div:
      -1 = bearish divergence, +1 = bullish divergence, 0 = none
    """
    price = df["close"]
    ph = price.rolling(lookback).max().shift(1)
    pl = price.rolling(lookback).min().shift(1)

    for length in lengths:
        col = f"RSI_{length}"
        div_col = f"{col}_div"
        if col in df.columns:
            rh = df[col].rolling(lookback).max().shift(1)
            rl = df[col].rolling(lookback).min().shift(1)

            bear = (price >= ph) & (df[col] < rh)
            bull = (price <= pl) & (df[col] > rl)

            df[div_col] = 0
            df.loc[bear, div_col] = -1
            df.loc[bull, div_col] = 1

    return df

def add_stochastic(df, k=14, d=3, smooth_k=3):
    """Append stochastic oscillator columns via pandas_ta."""
    df.ta.stoch(k=k, d=d, smooth_k=smooth_k, append=True)
    return df

def add_stoch_signals(df, stoch_col='STOCHk_14_3_3', ob_level=80, os_level=20):
    """
    Append stochastic overbought/oversold binary signals.
    """
    if stoch_col in df.columns:
        s = pd.to_numeric(df[stoch_col], errors='coerce')
        df[f'{stoch_col}_OB'] = (s > ob_level).astype(int)
        df[f'{stoch_col}_OS'] = (s < os_level).astype(int)
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    """Append MACD lines via pandas_ta."""
    df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
    return df

def add_macd_cross(df, macd='MACD_12_26_9', sig='MACDs_12_26_9'):
    """
    Append MACD cross signal: +1 bull cross, -1 bear cross, 0 none.
    """
    if macd in df.columns and sig in df.columns:
        m = pd.to_numeric(df[macd], errors='coerce')
        s = pd.to_numeric(df[sig], errors='coerce')
        up = (m > s) & (m.shift(1) < s.shift(1))
        dn = (m < s) & (m.shift(1) > s.shift(1))
        df['MACD_cross'] = np.where(up, 1, np.where(dn, -1, 0))
    return df

def add_ppo(df, fast=12, slow=26, signal=9):
    """Append PPO via pandas_ta."""
    df.ta.ppo(fast=fast, slow=slow, signal=signal, append=True)
    return df


def add_roc(df, length=10):
    """Append Rate of Change via pandas_ta."""
    df.ta.roc(length=length, append=True)
    return df