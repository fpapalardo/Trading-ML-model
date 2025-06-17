"""
Live Trading System with Real-Time SignalR Market Data

This module implements a live trading system that uses SignalR WebSockets
for real-time market data instead of polling. It processes 5-minute bars
as they complete and generates trading signals using a pre-trained model.

Key Features:
- Real-time bar processing via SignalR
- Efficient indicator computation with caching
- Thread-safe signal processing
- Automatic order management
- Trading hours enforcement

Author: Trading Bot
Date: 2024
"""

import os
import time
import traceback
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import numpy as np

from indicator_calculation import compute_all_indicators, session_times
from projectx_connector import ProjectXClient
from signalr_market_hub import TopStepMarketDataManager
from config import DATA_DIR, FUTURES


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Contract Settings
CONTRACT_SEARCH = "MNQ"  # Search term for finding contract
CONTRACT_ID = None  # Numeric ID for REST API (populated at runtime)
CONTRACT_SYMBOL = None  # String ID for SignalR (e.g., 'CON.F.US.NQ.H25')

# Data Settings
BAR_FILE = f"{DATA_DIR}/live/NQ/bar_data.csv"
LOOKBACK = 350  # Number of historical bars to maintain

# Trading Parameters
TP_ATR_MULTIPLIER = 2.0  # Take profit ATR multiplier
SL_ATR_MULTIPLIER = 1.5  # Stop loss ATR multiplier
TICK_SIZE = 0.25  # Minimum price increment

# Model Settings
MODEL_FILE = "rf_model_classifier_LOOKAHEAD_6_session_less.pkl"
FEATURE_COLUMNS = [
    'POC_Dist_Current_Points_1h', 'POC_Dist_Current_Points_5min', 'Day_of_Week', 
    'POC_Dist_Current_Points_15min', 'Day_Sin', 'RSI_7_5min', 'Minus_DI_14_1h', 
    'Trend_Score_15min', 'Trend_Strength_5min', 'Prev_Swing_Dist_15min', 'Time_Sin', 
    'Volume_Trend_15min', 'Is_Trending_5min', 'Is_Trending_15min'
]

# Timezone
NY_TZ = ZoneInfo("America/New_York")


# ═══════════════════════════════════════════════════════════════════════════
# TRADING STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class TradingState:
    """
    Manages the global state of the trading system.
    
    This class encapsulates all stateful components including the model,
    API connections, market data, and cached indicators.
    """
    
    def __init__(self):
        # Core components
        self.model = None
        self.px = None  # ProjectX API client
        self.market_hub = None  # SignalR market data
        
        # Indicator caches (avoid recomputation)
        self.f5_cache = None  # 5-minute indicators
        self.f15_cache = None  # 15-minute indicators
        self.f1h_cache = None  # 1-hour indicators
        
        # Timing tracking
        self.last_15m_ts = None
        self.last_1h_ts = None
        self.last_bar_timestamp = None
        
        # Data storage
        self.df_window = None  # DataFrame of historical bars
        self.bar_buffer = deque(maxlen=LOOKBACK)  # Real-time bar buffer
        
        # Thread safety
        self.processing_lock = threading.Lock()
        
    def initialize(self):
        """
        Initialize all components of the trading system.
        
        This method:
        1. Loads the ML model
        2. Authenticates with the broker API
        3. Finds and validates the contract
        4. Loads historical data
        5. Starts real-time data streaming
        """
        global CONTRACT_ID, CONTRACT_SYMBOL
        
        # Load trading model
        print("Loading model...")
        self.model = joblib.load(MODEL_FILE)
        
        # Initialize broker API
        print("Connecting to broker API...")
        self.px = ProjectXClient(
            FUTURES["topstep"]["username"], 
            FUTURES["topstep"]["api_key"]
        )
        self.px.authenticate(preferred_account_name="100KTC-V2-68606-92822961")
        
        # Find contract
        if CONTRACT_ID is None:
            print(f"Searching for contract: {CONTRACT_SEARCH}")
            contract_info = self.px.get_contract_info(CONTRACT_SEARCH)
            CONTRACT_ID = contract_info['id']  # Numeric ID for REST
            CONTRACT_SYMBOL = contract_info['id']  # String for SignalR
            CONTRACT_NAME = contract_info['name'] # Reference name
            print(f"Contract found: {CONTRACT_NAME} (ID: {CONTRACT_ID})")
        
        # Load historical data
        print("Loading historical bars...")
        self.df_window = self.load_initial_bars()
        print(f"Loaded {len(self.df_window)} bars")
        
        # Convert to buffer for real-time updates
        for _, row in self.df_window.iterrows():
            self.bar_buffer.append({
                't': row.name,
                'o': row['open'],
                'h': row['high'],
                'l': row['low'],
                'c': row['close'],
                'v': row['volume']
            })
        
        # Initialize SignalR connection
        print("Connecting to real-time market data...")
        self.market_hub = TopStepMarketDataManager(self.px.token)
        self.market_hub.add_bar_handler(self.on_new_bar)
        self.market_hub.start(CONTRACT_SYMBOL)
        
    def load_initial_bars(self) -> pd.DataFrame:
        """
        Load initial historical bars from the API.
        
        Returns:
            DataFrame with OHLCV data indexed by datetime
        """
        end_ny = datetime.now(NY_TZ).replace(second=0, microsecond=0)
        start_ny = end_ny - timedelta(minutes=LOOKBACK * 5)
        
        bars = self.px.get_bars(
            CONTRACT_ID,
            start_ny.astimezone(timezone.utc),
            end_ny.astimezone(timezone.utc),
            unit=2,  # Minutes
            unit_number=5,
            limit=LOOKBACK
        )
        
        if not bars:
            raise RuntimeError("No bars returned from API")
        
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        df['t'] = pd.to_datetime(df['t'], utc=True).dt.tz_convert(NY_TZ)
        df.rename(columns={
            't': 'datetime', 'o': 'open', 'h': 'high',
            'l': 'low', 'c': 'close', 'v': 'volume'
        }, inplace=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Save initial data
        os.makedirs(os.path.dirname(BAR_FILE), exist_ok=True)
        df.to_csv(BAR_FILE)
        
        return df
        
    def on_new_bar(self, bar_data: dict):
        """
        Callback for new bar from SignalR.
        
        This method is called whenever a 5-minute bar completes.
        It updates the data, saves to disk, and triggers signal processing.
        
        Args:
            bar_data: Dictionary with OHLCV data and timestamp
        """
        with self.processing_lock:
            try:
                # Parse bar timestamp
                bar_time = pd.to_datetime(bar_data['t']).tz_localize('UTC').tz_convert(NY_TZ)
                
                # Skip if not a new bar
                if self.last_bar_timestamp and bar_time <= self.last_bar_timestamp:
                    return
                
                print(f"\n[{datetime.now(NY_TZ):%H:%M:%S}] New 5min bar completed: {bar_time}")
                
                # Add to buffer
                self.bar_buffer.append({
                    't': bar_time,
                    'o': bar_data['o'],
                    'h': bar_data['h'],
                    'l': bar_data['l'],
                    'c': bar_data['c'],
                    'v': bar_data['v']
                })
                
                # Update DataFrame
                self.df_window = buffer_to_dataframe(self.bar_buffer)
                self.last_bar_timestamp = bar_time
                
                # Append to file
                append_bar_to_file(bar_data)
                
                # Process signal in separate thread
                threading.Thread(
                    target=process_new_bar_signal,
                    args=(self.df_window.copy(), bar_time),
                    daemon=True
                ).start()
                
            except Exception as e:
                print(f"[ERROR] Failed to process new bar: {e}")
                traceback.print_exc()


# Create global state instance
state = TradingState()


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def round_tick(price: float) -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Raw price value
        
    Returns:
        Price rounded to nearest tick
    """
    if price is None:
        return None
    return round(price * 4) * 0.25  # Fast tick rounding


def buffer_to_dataframe(buffer: deque) -> pd.DataFrame:
    """
    Convert bar buffer to pandas DataFrame.
    
    Args:
        buffer: Deque of bar dictionaries
        
    Returns:
        DataFrame with OHLCV data indexed by datetime
    """
    data = list(buffer)
    df = pd.DataFrame(data)
    df.set_index('t', inplace=True)
    df.index.name = 'datetime'
    df.rename(columns={
        'o': 'open', 'h': 'high',
        'l': 'low', 'c': 'close', 'v': 'volume'
    }, inplace=True)
    return df.sort_index()


def append_bar_to_file(bar_data: dict):
    """
    Append new bar to CSV file for persistence.
    
    Args:
        bar_data: Bar dictionary with OHLCV data
    """
    try:
        bar_time = pd.to_datetime(bar_data['t']).tz_localize('UTC').tz_convert(NY_TZ)
        with open(BAR_FILE, 'a') as f:
            f.write(f"{bar_time},{bar_data['o']},{bar_data['h']},{bar_data['l']},{bar_data['c']},{bar_data['v']}\n")
    except Exception as e:
        print(f"[WARN] Failed to append bar to file: {e}")


def check_trading_hours(now: datetime) -> bool:
    """
    Check if current time is within trading hours.
    
    Args:
        now: Current datetime in NY timezone
        
    Returns:
        True if within trading hours, False otherwise
    """
    # Skip 4 PM - 5 PM ET (market closed)
    return not (16 <= now.hour <= 17)


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_indicators_cached(df: pd.DataFrame, timeframe: str, features: list) -> pd.DataFrame:
    """
    Compute technical indicators with intelligent caching.
    
    This function avoids recomputing indicators when only new bars are added,
    significantly improving performance.
    
    Args:
        df: OHLCV DataFrame
        timeframe: Time period ('5min', '15min', '1h')
        features: List of indicator features to compute
        
    Returns:
        DataFrame with computed indicators
    """
    if timeframe == '5min':
        # For 5min, try incremental computation
        if state.f5_cache is not None and len(df) > len(state.f5_cache):
            new_rows = len(df) - len(state.f5_cache)
            if new_rows <= 5:  # Only if a few new bars
                # Compute indicators for recent data with context
                new_data = df.tail(50)  # Get enough context for indicators
                new_indicators = compute_all_indicators(new_data, suffix='_5min', features=features)
                new_indicators = session_times(new_indicators)
                # Append only new rows to cache
                result = pd.concat([state.f5_cache, new_indicators.tail(new_rows)])
                return result.tail(LOOKBACK)
    
    # Full computation for other timeframes or when cache is stale
    result = compute_all_indicators(df.copy(), suffix=f'_{timeframe}', features=features)
    if timeframe == '5min':
        result = session_times(result)
    return result


def is_new_timeframe_bar(now: datetime, last_ts: datetime, minutes: int) -> bool:
    """
    Check if we need to compute new indicators for a timeframe.
    
    Args:
        now: Current time
        last_ts: Last computation timestamp
        minutes: Timeframe in minutes
        
    Returns:
        True if new computation needed
    """
    if last_ts is None:
        return True
    return (now - last_ts).total_seconds() >= minutes * 60


def prepare_features_realtime(df_window: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """
    Prepare feature set for model prediction.
    
    This function computes all required indicators across multiple timeframes
    and merges them into a single feature vector.
    
    Args:
        df_window: Historical OHLCV data
        now: Current timestamp
        
    Returns:
        DataFrame with one row containing all features
    """
    # Always compute 5min indicators (base timeframe)
    state.f5_cache = compute_indicators_cached(df_window, '5min', ['all'])
    
    # Compute 15min indicators when needed
    if is_new_timeframe_bar(now, state.last_15m_ts, 15):
        df15 = df_window.resample('15min', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        state.f15_cache = compute_all_indicators(
            df15.copy(), 
            suffix='_15min', 
            features=['volume_trend', 'prev_swing', 'trend', 'poc', 'adx', 'ema', 'atr']
        )
        state.last_15m_ts = now
    
    # Compute 1h indicators when needed
    if is_new_timeframe_bar(now, state.last_1h_ts, 60):
        df1h = df_window.resample('1h', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        state.f1h_cache = compute_all_indicators(
            df1h.copy(), 
            suffix='_1h', 
            features=['adx', 'poc']
        )
        state.last_1h_ts = now
    
    # Merge all timeframes into final feature set
    if state.f15_cache is not None and state.f1h_cache is not None:
        # Get latest 5min row
        f5_last = state.f5_cache.iloc[[-1]].copy()
        
        # Get latest values from higher timeframes
        f15_latest = state.f15_cache.filter(regex='_15min').iloc[[-1]]
        f1h_latest = state.f1h_cache.filter(regex='_1h').iloc[[-1]]
        
        # Merge all features
        for col in f15_latest.columns:
            f5_last[col] = f15_latest[col].iloc[0]
        for col in f1h_latest.columns:
            f5_last[col] = f1h_latest[col].iloc[0]
            
        return f5_last
    
    # Return whatever we have if higher timeframes not ready
    return state.f5_cache.tail(1) if state.f5_cache is not None else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING AND ORDER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def process_new_bar_signal(df_window: pd.DataFrame, bar_time: datetime):
    """
    Process trading signal for a newly completed bar.
    
    This function:
    1. Checks trading hours
    2. Verifies no existing orders
    3. Prepares features
    4. Generates prediction
    5. Places order if signal present
    
    Args:
        df_window: Historical OHLCV data
        bar_time: Timestamp of the completed bar
    """
    now = datetime.now(NY_TZ)
    
    # Check trading hours
    if not check_trading_hours(now):
        # Cancel any open orders outside trading hours
        try:
            open_orders = state.px.search_open_orders()
            for order in open_orders:
                oid = order.get('orderId') or order.get('id')
                state.px.cancel_order(oid)
                print(f"[{now}] Canceled after-hours order {oid}")
        except Exception as e:
            print(f"[WARN] Failed to cancel after-hours orders: {e}")
        return

    # Check for existing open orders
    try:
        if state.px.search_open_orders():
            print("Skipping: existing open orders detected")
            return
    except Exception as e:
        print(f"[WARN] Failed to check open orders: {e}")
        return

    # Prepare features
    features_df = prepare_features_realtime(df_window, now)
    if features_df.empty:
        print("Features not ready, skipping signal")
        return
        
    # Verify all required features present
    missing_features = set(FEATURE_COLUMNS) - set(features_df.columns)
    if missing_features:
        print(f"Missing features: {missing_features}")
        return

    # Extract feature vector
    X = features_df[FEATURE_COLUMNS].iloc[[-1]]
    
    # Get current price and ATR for position sizing
    atr = features_df['ATR_14_5min'].iat[-1]
    price = df_window['close'].iat[-1]
    
    # Generate prediction
    pred = state.model.predict(X)[0]
    
    # Map prediction to action
    # Assuming: 0 = no signal, 1 = buy, 2 = sell
    if pred not in (1, 2):
        print(f"No trade signal (prediction={pred}) for bar {bar_time}")
        return

    # Determine trade direction and calculate TP/SL
    if pred == 1:  # Buy signal
        side = 'Buy'
        tp_price = round_tick(price + (TP_ATR_MULTIPLIER * atr))
        sl_price = round_tick(price - (SL_ATR_MULTIPLIER * atr))
    else:  # Sell signal
        side = 'Sell'
        tp_price = round_tick(price - (TP_ATR_MULTIPLIER * atr))
        sl_price = round_tick(price + (SL_ATR_MULTIPLIER * atr))

    # Place order
    try:
        state.px.place_order(
            CONTRACT_ID, 
            side, 
            quantity=1, 
            limit_price=tp_price, 
            stop_price=sl_price
        )
        
        print(f"[{now}] {side} signal @ {price:.2f}")
        print(f"  TP: {tp_price:.2f} (+{abs(tp_price - price):.2f} pts)")
        print(f"  SL: {sl_price:.2f} (-{abs(sl_price - price):.2f} pts)")
        print(f"  Bar: {bar_time}")
        
    except Exception as e:
        print(f"[ERROR] Order placement failed: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution loop for the live trading system.
    
    This function:
    1. Initializes all components
    2. Starts real-time data streaming
    3. Maintains connection health
    4. Handles graceful shutdown
    """
    print("=" * 70)
    print("LIVE TRADING SYSTEM WITH SIGNALR")
    print("=" * 70)
    
    try:
        # Initialize everything
        state.initialize()
        print(f"\n✓ System initialized successfully")
        print(f"✓ Contract: {CONTRACT_SYMBOL}")
        print(f"✓ Historical bars: {len(state.df_window)}")
        print(f"✓ Real-time data: Connected")
        print("\nWaiting for market data...\n")
        
        # Show initial market quote
        time.sleep(2)
        latest_quote = state.market_hub.get_latest_quote(CONTRACT_SYMBOL)
        if latest_quote:
            bid = latest_quote.get('bid', 'N/A')
            ask = latest_quote.get('ask', 'N/A')
            print(f"Current market: Bid={bid} Ask={ask}")
        
    except Exception as e:
        print(f"\n[FATAL] Initialization failed: {e}")
        traceback.print_exc()
        return

    # Main loop - just keeps the program running
    # All actual work happens in SignalR callbacks
    try:
        print("\nSystem running. Press Ctrl+C to stop.\n")
        
        while True:
            time.sleep(10)
            
            # Periodic health check
            if not state.market_hub.market_hub.is_connected:
                print(f"\n[{datetime.now(NY_TZ):%H:%M:%S}] SignalR disconnected, reconnecting...")
                if state.market_hub.market_hub.connect():
                    print("Reconnected successfully")
                else:
                    print("Reconnection failed")
            
            # Optional: Show market heartbeat every minute
            if datetime.now().second < 5:
                quote = state.market_hub.get_latest_quote(CONTRACT_SYMBOL)
                if quote:
                    bid = quote.get('bid', 'N/A')
                    ask = quote.get('ask', 'N/A')
                    spread = float(ask) - float(bid) if bid != 'N/A' and ask != 'N/A' else 'N/A'
                    print(f"[{datetime.now(NY_TZ):%H:%M:%S}] Heartbeat: Bid={bid} Ask={ask} Spread={spread}")
                else:
                    print(f"No quotes for {CONTRACT_SYMBOL}")
                
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        
        # Graceful shutdown
        if state.market_hub:
            print("- Disconnecting market data...")
            state.market_hub.stop()
            
        print("- Shutdown complete")
        
    except Exception as e:
        print(f"\n[FATAL] Unexpected error: {e}")
        traceback.print_exc()
        
        # Emergency shutdown
        if state.market_hub:
            state.market_hub.stop()


if __name__ == "__main__":
    main()