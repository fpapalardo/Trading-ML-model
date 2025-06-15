import pandas as pd
import numpy as np
import pandas_ta as ta

# --- Volatility Indicators & Features ---
def add_atrs_all(
    df: pd.DataFrame,
    lengths: tuple[int, ...] = (3, 5, 7, 14)
) -> pd.DataFrame:
    """
    Compute ATR for each period in `lengths` and append as ATR_<length>.
    """
    for length in lengths:
        col = f"ATR_{length}"
        # force pandas_ta to name it exactly ATR_<length>
        df.ta.atr(length=length, append=True, col_names=(col,))
    return df

def add_bollinger(df, length=20, std=2):
    """Append Bollinger Bands via pandas_ta."""
    df.ta.bbands(length=length, std=std, append=True)
    return df

def add_choppiness(df, length=14):
    """Append Choppiness Index via pandas_ta."""
    df.ta.chop(length=length, append=True)
    return df

def rolling_stats(df, price_col='close', window1=14, window2=30):
    """Append rolling std, skew, kurtosis of returns."""
    ret = df[price_col].pct_change().replace([np.inf, -np.inf], np.nan)
    df[f'Rolling_Std_{window1}'] = ret.rolling(window1).std()
    df[f'Rolling_Skew_{window2}'] = ret.rolling(window2).skew()
    df[f'Rolling_Kurt_{window2}'] = ret.rolling(window2).kurt()
    return df

def daily_vwap(df, high='high', low='low', close='close', volume='volume', new_col='VWAP_D'):
    """Append daily VWAP."""
    temp = df.copy()
    for col in [high, low, close, volume]:
        temp[col] = pd.to_numeric(temp[col], errors='coerce')
    tpv = ((temp[high]+temp[low]+temp[close])/3)*temp[volume]
    cum_tpv = tpv.groupby(temp.index.date).cumsum()
    cum_vol = temp[volume].groupby(temp.index.date).cumsum()
    df[new_col] = (cum_tpv/cum_vol).replace([np.inf, -np.inf], np.nan)
    return df

def price_vs_bb(df, price='close', bb_up='BBU_20_2.0', bb_lo='BBL_20_2.0'):
    """Append whether price is above/below Bollinger bands."""
    if price in df.columns and bb_up in df.columns and bb_lo in df.columns:
        p = df[price]; up = df[bb_up]; lo = df[bb_lo]
        df['Price_vs_BB_Up'] = (p > up).astype(int)
        df['Price_vs_BB_Lo'] = (p < lo).astype(int)
    return df

def add_cci(df, length=20):
    """Append Commodity Channel Index via pandas_ta."""
    df.ta.cci(length=length, append=True)
    return df


def add_willr(df, length=14):
    """Append Williams %R via pandas_ta."""
    df.ta.willr(length=length, append=True)
    return df


def mean_reversion(df, bbm='BBM_20_2.0', atr='ATR_14', window=10):
    """Append mean-reversion potential (price vs BB mid normalized by ATR)."""
    if bbm in df.columns and atr in df.columns:
        df['Mean_Reversion'] = ((df['close'] - df[bbm]) / df[atr]).rolling(window).mean()
    return df