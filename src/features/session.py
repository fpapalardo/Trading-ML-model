import pandas as pd
import numpy as np

# --- Session & Time Features ---
def session_id(df):
    """Integer session key (UTC->NY shift then date)."""
    sid = (df.index - pd.Timedelta(hours=18)).normalize()
    df['session_id'] = sid.view(np.int64)
    return df

def session_range(df):
    """Append session range and price_vs_session_range."""
    d_str = df.index.normalize().strftime('%Y-%m-%d')
    hr = df.index.hour
    sess = pd.Series(np.select(
        [(hr<6),(hr<13),(hr<20)],
        ['asia','london','new_york'],
        default='overnight'
    ), index=df.index)
    key = sess+'_'+d_str
    hi = df.groupby(key)['high'].transform('max')
    lo = df.groupby(key)['low'].transform('min')
    df['Price_vs_Sess'] = (df['close']-lo)/(hi-lo+1e-6)
    df['session']=sess
    return df

def time_session_features(df):
    """Append cyclical time and session flags."""
    hr, mn, dw = df.index.hour, df.index.minute, df.index.dayofweek
    tod = hr+mn/60
    df['Time_Sin']=np.sin(2*np.pi*tod/24)
    df['Time_Cos']=np.cos(2*np.pi*tod/24)
    df['Day_Sin']=np.sin(2*np.pi*dw/7)
    df['Day_Cos']=np.cos(2*np.pi*dw/7)
    df['Is_NY'] = ((hr>=8)&(hr<17)).astype(int)
    return df