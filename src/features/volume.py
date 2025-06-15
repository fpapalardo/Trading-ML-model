import pandas as pd
import numpy as np
import pandas_ta as ta

# --- Volume Indicators & Features ---
def add_obv(df):
    """Append OBV via pandas_ta."""
    df.ta.obv(append=True)
    return df

def volume_spike(df, window=20, mult=2):
    """Append volume spike flag."""
    m = df['volume'].rolling(window).mean()
    df['Vol_Spike'] = (df['volume']>mult*m).astype(int)
    return df

def donchian_dist(df, length=20):
    hi = df['high'].rolling(length).max(); lo = df['low'].rolling(length).min()
    mid = (hi+lo)/2
    df['Donchian_Dist'] = (df['close']-mid)/df['ATR_14']
    return df

def volume_delta_features(df):
    df['Vol_Delta_1'] = df['volume'].diff()
    df['Vol_Delta_2'] = df['volume'].diff(2)
    return df

def volume_zscore_features(df, window=20):
    m = df['volume'].rolling(window).mean(); s = df['volume'].rolling(window).std()
    df['Vol_Zscore'] = (df['volume']-m)/s
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session-based VWAP, POC, and related volume measures.

    Adds:
      - session_id                 : Timestamp floored to session “day”
      - VWAP_Session               : cumulative TPV / cumulative vol per session
      - POC_Current                : per-session price level with max volume
      - POC_Previous               : prior-session’s POC_Current
      - POC_Dist_Current_Points    : close - POC_Current
    """
    # identify session “day” by shifting back 18h then flooring to midnight
    df = df.copy()
    df['session_id'] = (df.index - pd.Timedelta(hours=18)).floor('D')

    # VWAP per session
    tpv = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    cum_tpv = tpv.groupby(df['session_id']).cumsum()
    cum_vol = df['volume'].groupby(df['session_id']).cumsum()
    df['VWAP_Session'] = cum_tpv / cum_vol

    # compute current-session POC (price level with max volume)
    prof = (
        df
        .groupby(['session_id', df['close']])['volume']
        .sum()
        .reset_index(name='vol_sum')
    )
    poc_map = (
        prof
        .sort_values(['session_id','vol_sum'], ascending=[True,False])
        .drop_duplicates('session_id')
        .set_index('session_id')['close']
    )
    df['POC_Current'] = df['session_id'].map(poc_map)

    # map prior-session POC
    # build an ordered list of sessions, then shift by 1
    sessions = sorted(poc_map.index)
    prev_map = {curr: prev for prev, curr in zip(sessions, sessions[1:])}
    df['session_prev'] = df['session_id'].map(prev_map)
    df['POC_Previous'] = df['session_prev'].map(poc_map)

    # distance from current POC in points
    df['POC_Dist_Current_Points'] = df['close'] - df['POC_Current']

    return df

def add_volume_trend(
    df: pd.DataFrame,
    window: int = 20,
    new_col: str = "Volume_Trend"
) -> pd.DataFrame:
    """
    Add a rolling volume trend feature:
      Volume_Trend = (volume / rolling_mean(volume, window) - 1) * 100
    Only adds the column if it does not already exist.
    """
    df = df.copy()
    
    # Skip if already present
    if new_col in df.columns:
        return df
    
    # Ensure volume exists
    if "volume" not in df.columns:
        raise KeyError("Column 'volume' is required for add_volume_trend.")
    
    vol = pd.to_numeric(df["volume"], errors="coerce")
    mv  = vol.rolling(window).mean()
    df[new_col] = (vol / mv - 1) * 100
    
    return df
