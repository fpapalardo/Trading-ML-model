import pandas as pd
import numpy as np

# --- Price Action & Patterns ---
def candle_features(df):
    """Append wick, body, and related metrics."""
    h, l, o, c = df['high'], df['low'], df['open'], df['close']
    df['Candle_Range'] = h-l
    df['Candle_Body'] = (c-o).abs()
    df['Upper_Wick'] = h-np.maximum(o,c)
    df['Lower_Wick'] = np.minimum(o,c)-l
    return df

def return_features(df):
    """Append log and simple returns."""
    p = pd.to_numeric(df['close'], errors='coerce').replace(0,np.nan)
    df['Log_Ret_1'] = np.log(p/p.shift(1))
    df['Simple_Ret_1'] = p.pct_change()
    return df

def prev_swing_high_low(df, window=12):
    df['Prev_Sw_High'] = df['high'].rolling(window).max().shift(1)
    df['Prev_Sw_Low']  = df['low'].rolling(window).min().shift(1)
    return df

def dist_to_closest_sr(df):
    df['Dist_High'] = (df['Prev_Sw_High']-df['close']).abs()
    df['Dist_Low']  = (df['Prev_Sw_Low']-df['close']).abs()
    df['Dist_SR']   = df[['Dist_High','Dist_Low']].min(axis=1)
    return df

def candlestick_patterns(df):
    df['Bull_Engulf'] = ((df['close']>df['open'])&(df['open']<df['close'].shift(1))&(df['close']>df['open'].shift(1))).astype(int)
    df['Bear_Engulf'] = ((df['close']<df['open'])&(df['open']>df['close'].shift(1))&(df['close']<df['open'].shift(1))).astype(int)
    return df

def stop_hunt(df):
    upper = df['high']-np.maximum(df['open'],df['close'])
    lower = np.minimum(df['open'],df['close'])-df['low']
    rng = df['high']-df['low']+1e-6
    df['Stop_Hunt'] = ((upper>0.6*rng)|(lower>0.6*rng)).astype(int)
    return df

def fvg(df):
    h1, l1 = df['high'].shift(1), df['low'].shift(1)
    h2, l2 = df['high'].shift(-1),df['low'].shift(-1)
    bg = l2>h1; br = h2<l1
    df['FVG_Exists'] = (bg|br).astype(int)
    size = pd.Series(0., index=df.index)
    size[bg]= (l2-h1)[bg]; size[br]=(l1-h2)[br]
    df['FVG_Size']=size
    pos = pd.Series(0, index=df.index)
    pos[bg&(df['close']>l2)] = 1
    pos[br&(df['close']<h2)] = -1
    df['FVG_Pos']=pos
    return df

def day_high_low_open(df):
    d = df.index.date
    df['Open_Day']=df.groupby(d)['open'].transform('first')
    df['High_Day']=df.groupby(d)['high'].transform('max')
    df['Low_Day']=df.groupby(d)['low'].transform('min')
    return df

def prev_high_low(df):
    df['Prev_High']=df['high'].shift(1)
    df['Prev_Low']=df['low'].shift(1)
    return df

def price_vs_open(df):
    df['Price_vs_Open'] = df['close'] - df['Open_Day']
    return df