import pandas as pd
import numpy as np

# --- Trend Indicators & Features ---
def add_ema(df, length=20):
    """Append EMA via pandas_ta."""
    df.ta.ema(length=length, append=True)
    return df

def add_sma(df, length=20):
    """Append SMA via pandas_ta."""
    df.ta.sma(length=length, append=True)
    return df

def add_adx(df, length=14):
    """Append ADX via pandas_ta."""
    df.ta.adx(length=length, append=True)
    return df

def add_trend_features(df):
    """
    Composite trend features: direction, strength, price vs EMA/VWAP, regime, score.
    """
    # directional
    td = np.where(df['EMA_20']>df['EMA_50'], 1, -1)
    ts = ((df['EMA_20']-df['EMA_50'])/df['ATR_14']).rolling(20).mean()
    pve = ((df['close']-df['EMA_20'])/df['EMA_20'])*100
    pvv = ((df['close']-df['VWAP_D'])/df['VWAP_D'])*100
    df['Trend_Direction']   = td
    df['Trend_Strength']    = ts
    df['Price_vs_EMA20']    = pve
    df['Price_vs_VWAP']     = pvv
    df['Trend_Score']       = (td*20 + ts*10 + (df['RSI_14']-50)).clip(-100,100)
    return df

def ema_trend_confirmation(df):
    """Append EMA trend confirmation: 1 bull, -1 bear, 0 none."""
    e9, e21, e50 = df['EMA_9'], df['EMA_21'], df['EMA_50']
    slope = e21 - e21.shift()
    bt = (e9>e21)&(e21>e50)&(slope>0)
    br = (e9<e21)&(e21<e50)&(slope<0)
    df['EMA_Trend_Conf'] = 0
    df.loc[bt,'EMA_Trend_Conf']=1
    df.loc[br,'EMA_Trend_Conf']=-1
    return df

def add_market_regime_features(df):
    """Append market regime flags and combined label."""
    adx = df['ADX_14']; chop = df['CHOP_14_1_100']; atr = df['ATR_14']
    thr = atr.rolling(100,min_periods=20).mean()*1.1
    df['Is_Trending'] = (adx>20).astype(int)
    df['Is_Choppy']   = (chop>60).astype(int)
    df['Is_High_Vol'] = (atr>thr).astype(int)
    df['Regime']      = df['Is_Trending'].astype(str)+'_'+df['Is_High_Vol'].astype(str)
    return df