import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings

# --- Volume Indicators & Features ---
def add_obv(df):
    """Append OBV via pandas_ta."""
    df.ta.obv(append=True)
    return df

def add_obv_relative_features(df, lookback=100):
    """Add relative OBV features that don't depend on absolute values"""
    # Calculate traditional OBV
    df.ta.obv(append=True)
    
    # Add relative features
    df['OBV_zscore'] = (df['OBV'] - df['OBV'].rolling(lookback).mean()) / df['OBV'].rolling(lookback).std()
    df['OBV_percentile'] = df['OBV'].rolling(lookback).rank(pct=True)
    df['OBV_momentum'] = df['OBV'].diff(20)  # 20-period momentum
    df['OBV_acceleration'] = df['OBV'].diff().diff()  # Second derivative
    
    # Drop the absolute OBV
    df = df.drop(columns=['OBV'])
    
    return df

def add_obv_session_based(df):
    """Calculate OBV that resets each session (6 PM ET)"""
    df = df.copy()
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    df['session_id'] = (df.index - pd.Timedelta(hours=18)).floor('D')
    
    def calc_obv(group):
        price_diff = group['close'].diff()
        volume_direction = price_diff.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        obv = (volume_direction * group['volume']).cumsum()
        return obv
    
    # FIX: Explicitly select columns for the apply function
    df['OBV_Session'] = df.groupby('session_id')[['close', 'volume']].apply(calc_obv).reset_index(level=0, drop=True)
    
    df['OBV_Session_Change'] = df['OBV_Session'].diff()
    df['OBV_Session_ROC'] = df['OBV_Session'].pct_change(periods=12)
    
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
    Session-based VWAP, POC, and related volume measures, calculated
    without lookahead bias. (Final corrected version)
    """
    df = df.copy()
    df['session_id'] = (df.index - pd.Timedelta(hours=18)).floor('D')

    # VWAP calculation
    tpv = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    cum_tpv = tpv.groupby(df['session_id']).cumsum()
    cum_vol = df['volume'].groupby(df['session_id']).cumsum()
    df['VWAP_Session'] = (cum_tpv / cum_vol).ffill()

    # --- FINAL FIX: Point-in-Time POC Calculation ---

    # We'll build a list of point-in-time POC values for each session
    all_poc_values = []

    # Group by session and iterate through each session's data
    for _, session_df in df.groupby('session_id'):
        # Create an expanding window object on the session data
        expanding_window = session_df.expanding()
        
        # This list will hold POC values for the current session
        session_pocs = []
        
        # Iterate through the expanding window. For each step, 'window' will
        # be the DataFrame from the start of the session up to the current row.
        for window in expanding_window:
            # Group the current window of data by price and sum the volume
            volume_at_price = window.groupby('close')['volume'].sum()
            
            if not volume_at_price.empty:
                # Get the price (index) with the maximum volume
                current_poc = volume_at_price.idxmax()
                session_pocs.append(current_poc)
            else:
                session_pocs.append(np.nan) # Handle the very first empty window
        
        # Add the calculated POCs for this session to our main list
        all_poc_values.extend(session_pocs)

    # Assign the calculated POCs to the DataFrame
    df['POC_Current'] = all_poc_values
    df['POC_Current'] = df.groupby('session_id')['POC_Current'].ffill()

    # --- Previous POC and Distance Calculations ---
    final_poc_map = df.groupby('session_id')['POC_Current'].last()

    sessions = sorted(final_poc_map.index)
    prev_map = {curr: prev for prev, curr in zip(sessions, sessions[1:])}
    df['session_prev'] = df['session_id'].map(prev_map)
    df['POC_Previous'] = df['session_prev'].map(final_poc_map)
    df['POC_Previous'] = df['POC_Previous'].ffill() # Ensure previous POC is carried forward

    df['POC_Dist_Current_Points'] = df['close'] - df['POC_Current']

    # Clean up helper columns
    df = df.drop(columns=['session_id', 'session_prev'], errors='ignore')

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
