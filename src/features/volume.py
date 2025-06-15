import pandas as pd
import numpy as np

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


def add_volume_features(df):
    """Session-based VWAP, POC, and related vol measures."""
    df['session_id'] = ((df.index - pd.Timedelta(hours=18)).floor('D'))
    tpv = ((df['high']+df['low']+df['close'])/3)*df['volume']
    df['VWAP_Session'] = tpv.groupby(df['session_id']).cumsum()/df['volume'].groupby(df['session_id']).cumsum()
    # POC
    prof = df.groupby(['session_id', df['close']])['volume'].sum().reset_index()
    poc = prof.sort_values(['session_id','volume'], ascending=[True,False]).drop_duplicates('session_id').set_index('session_id')['close']
    df['POC_Current'] = df['session_id'].map(poc)
    return df