import pandas as pd

def price_vs_ma(df, price_col='close', ma_col='EMA_20', suffix='_vs_EMA20'):
    if price_col in df and ma_col in df:
        df[price_col+suffix] = pd.to_numeric(df[price_col]) / pd.to_numeric(df[ma_col])
    return df

def ma_vs_ma(df, ma1='EMA_10', ma2='EMA_20', suffix='_vs_EMA20'):
    if ma1 in df and ma2 in df:
        df[ma1+suffix] = pd.to_numeric(df[ma1]) / pd.to_numeric(df[ma2])
    return df

def ma_slope(df, ma='EMA_10', suffix='_Slope_10', periods=1):
    if ma in df:
        df[f'{ma}{suffix}'] = pd.to_numeric(df[ma]).diff(periods)/periods
    return df

def lagged_features(df, cols, lags=(1,3,6)):
    for c in cols:
        if c in df:
            s = pd.to_numeric(df[c])
            for lag in lags:
                df[f'{c}_Lag_{lag}'] = s.shift(lag)
    return df
