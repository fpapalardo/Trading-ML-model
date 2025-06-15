from .indicators    import price_vs_ma, ma_vs_ma, ma_slope, lagged_features
from .momentum      import (
    add_rsi, add_rsi_signals, rsi_divergence_feature,
    add_stochastic, add_stoch_signals, add_macd, add_macd_cross,
    add_ppo, add_roc,
)
from .volatility    import (
    add_atr, add_bollinger, daily_vwap, price_vs_bb,
    rolling_stats, add_choppiness,
    add_cci, add_willr, mean_reversion,
)
from .trend         import (
    add_ema, add_sma, add_adx,
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

FEATURE_FUNCTIONS = {
    # Indicators
    'price_vs_ma': price_vs_ma,
    'ma_vs_ma': ma_vs_ma,
    'ma_slope': ma_slope,
    'lagged': lagged_features,

    # Momentum
    'rsi': add_rsi,
    'rsi_signals': add_rsi_signals,
    'rsi_divergence': rsi_divergence_feature,
    'stochastic': add_stochastic,
    'stoch_signals': add_stoch_signals,
    'macd': add_macd,
    'macd_cross': add_macd_cross,
    'ppo': add_ppo,
    'roc': add_roc,

    # Volatility
    'atr': add_atr,
    'bollinger': add_bollinger,
    'daily_vwap': daily_vwap,
    'price_vs_bb': price_vs_bb,
    'rolling_stats': rolling_stats,
    'choppiness': add_choppiness,
    'cci': add_cci,
    'willr': add_willr,
    'mean_reversion': mean_reversion,

    # Trend
    'ema': add_ema,
    'sma': add_sma,
    'adx': add_adx,
    'trend_features': add_trend_features,
    'ema_trend_conf': ema_trend_confirmation,
    'market_regime': add_market_regime_features,

    # Volume
    'obv': add_obv,
    'volume_spike': volume_spike,
    'donchian_dist': donchian_dist,
    'volume_delta': volume_delta_features,
    'volume_zscore': volume_zscore_features,
    'volume_features': add_volume_features,

    # Volume stats
    'vol_delta_rollsum': add_volume_delta_rollsum,
    'cvd': add_cvd,
    'wick_pct': add_wick_percent,
    'rel_vol': add_rel_volume,

    # Price Action
    'candle': candle_features,
    'returns': return_features,
    'prev_swing': prev_swing_high_low,
    'dist_to_sr': dist_to_closest_sr,
    'patterns': candlestick_patterns,
    'stop_hunt': stop_hunt,
    'fvg': fvg,
    'day_high_low': day_high_low_open,
    'prev_high_low': prev_high_low,
    'price_vs_open': price_vs_open,

    # Session/Time
    'time_session': time_session_features,
    'session_id': session_id,
    'session_range': session_range,
}