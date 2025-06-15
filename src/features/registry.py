# src/features/registry.py

# ── Imports ──────────────────────────────────────────────────────────
from .indicators    import price_vs_all_mas, ma_vs_all_mas, ma_slope_all_mas, lagged_features
from .momentum      import (
    add_rsi_all, add_rsi_signals_all, rsi_divergence_all,
    add_stochastic, add_stoch_signals, add_macd, add_macd_cross,
    add_ppo, add_roc,
)
from .volatility    import (
    add_atrs_all, add_bollinger, daily_vwap, price_vs_bb,
    rolling_stats, add_choppiness,
    add_cci, add_willr, mean_reversion,
)
from .trend         import (
    add_emas_all, add_smas_all, add_adx,
    add_trend_features, ema_trend_confirmation,
    add_market_regime_features,
)
from .volume        import (
    add_obv, volume_spike, donchian_dist,
    volume_delta_features, volume_zscore_features,
    add_volume_features,
)
from .volume_stats  import (
    add_volume_delta_rollsum, add_cvd, add_wick_percent, add_rel_volume,
)
from .price_action  import (
    candle_features, return_features,
    prev_swing_high_low, dist_to_closest_sr,
    candlestick_patterns, stop_hunt, fvg,
    day_high_low_open, prev_high_low, price_vs_open,
)
from .session       import time_session_features, session_id, session_range

# ── Feature Functions ─────────────────────────────────────────────────
FEATURE_FUNCTIONS = {
    # Indicators
    'price_vs_ma':      price_vs_all_mas,
    'ma_vs_ma':         ma_vs_all_mas,
    'ma_slope':         ma_slope_all_mas,
    'lagged':           lagged_features,

    # Momentum
    "rsi":              add_rsi_all,           # computes RSI_7 & RSI_14
    "rsi_signals":      add_rsi_signals_all,   # OB/OS for both
    "rsi_divergence":   rsi_divergence_all,   # divergence for both
    'stochastic':       add_stochastic,
    'stoch_signals':    add_stoch_signals,
    'macd':             add_macd,
    'macd_cross':       add_macd_cross,
    'ppo':              add_ppo,
    'roc':              add_roc,

    # Volatility
    'atr':              add_atrs_all,
    'bollinger':        add_bollinger,
    'daily_vwap':       daily_vwap,
    'price_vs_bb':      price_vs_bb,
    'rolling_stats':    rolling_stats,
    'choppiness':       add_choppiness,
    'cci':              add_cci,
    'willr':            add_willr,
    'mean_reversion':   mean_reversion,

    # Trend
    "ema":              add_emas_all,
    "sma":              add_smas_all,
    'adx':              add_adx,
    'trend_features':   add_trend_features,
    'ema_trend_conf':   ema_trend_confirmation,
    'market_regime':    add_market_regime_features,

    # Volume
    'obv':              add_obv,
    'volume_spike':     volume_spike,
    'donchian_dist':    donchian_dist,
    'volume_delta':     volume_delta_features,
    'volume_zscore':    volume_zscore_features,
    'volume_features':  add_volume_features,

    # Volume Stats
    'vol_delta_rollsum': add_volume_delta_rollsum,
    'cvd':               add_cvd,
    'wick_pct':          add_wick_percent,
    'rel_vol':           add_rel_volume,

    # Price Action
    'candle':           candle_features,
    'returns':          return_features,
    'prev_swing':       prev_swing_high_low,
    'dist_to_sr':       dist_to_closest_sr,
    'patterns':         candlestick_patterns,
    'stop_hunt':        stop_hunt,
    'fvg':              fvg,
    'day_high_low':     day_high_low_open,
    'prev_high_low':    prev_high_low,
    'price_vs_open':    price_vs_open,
}

# ── Session / Time Functions ───────────────────────────────────────────
SESSION_FUNCTIONS = {
    'time_session':     time_session_features,
    'session_id':       session_id,
    'session_range':    session_range,
}
