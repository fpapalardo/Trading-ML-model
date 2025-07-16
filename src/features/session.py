import pandas as pd
import numpy as np

# --- Session & Time Features ---
def session_id(df):
    """Integer session key (UTC->NY shift then date)."""
    sid = (df.index - pd.Timedelta(hours=18)).normalize()
    df['session_id'] = sid.view(np.int64)
    return df

def session_range(df):
    """
    Append session range and price_vs_session_range without lookahead bias.
    """
    df = df.copy()
    d_str = df.index.normalize().strftime('%Y-%m-%d')
    hr = df.index.hour
    
    # Create a key for each session (e.g., 'asia_2025-07-06')
    sess = pd.Series(np.select(
        [(hr < 6), (hr < 13), (hr < 20)],
        ['asia', 'london', 'new_york'],
        default='overnight'
    ), index=df.index)
    key = sess + '_' + d_str
    
    # Group by the unique session key
    session_groups = df.groupby(key)
    
    # Calculate the expanding high and low for each session
    hi = session_groups['high'].expanding().max()
    lo = session_groups['low'].expanding().min()
    
    # The result has a multi-index, so we need to remove the group's index
    hi = hi.droplevel(0)
    lo = lo.droplevel(0)
    
    # Calculate price vs. session range using the point-in-time values
    df['Price_vs_Sess'] = (df['close'] - lo) / (hi - lo + 1e-6)
    df['session'] = sess
    
    return df

def time_session_features(df):
    """Append cyclical time and session flags."""
    hr, mn, dw = df.index.hour, df.index.minute, df.index.dayofweek
    tod = hr+mn/60
    df['Day_of_Week'] = df.index.dayofweek
    df['Time_Sin']=np.sin(2*np.pi*tod/24)
    df['Time_Cos']=np.cos(2*np.pi*tod/24)
    df['Day_Sin']=np.sin(2*np.pi*dw/7)
    df['Day_Cos']=np.cos(2*np.pi*dw/7)
    df['Is_NY'] = ((hr>=8)&(hr<17)).astype(int)
    return df