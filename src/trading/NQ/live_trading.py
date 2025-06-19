"""
Enhanced Live Trading System

This enhanced version includes:
- Auto-reconnection for SignalR
- OCO (One Cancels Other) order management
- Telegram notifications
- Better error handling and recovery

Key improvements:
1. Automatic reconnection when SignalR disconnects
2. Proper order lifecycle management with TP/SL cleanup
3. Real-time notifications via Telegram
4. Integration with enhanced trading monitor
"""

import os
import time
import traceback
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pytz

import joblib
import pandas as pd
import numpy as np
import asyncio

from indicator_calculation import compute_all_indicators, session_times
from projectx_connector import ProjectXClient
from signalr_user_hub import UserHubClient
from signalr_market_hub import TopStepMarketDataManager
from order_manager import OrderManager, place_bracket_order
from telegram_notifier import create_telegram_notifier, setup_telegram_notifications
from config import DATA_DIR, FUTURES


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Contract Settings
CONTRACT_SEARCH = "MNQ"  # Search term for finding contract
CONTRACT_ID = None  # Will be populated at runtime
CONTRACT_SYMBOL = None  # Will be populated at runtime

# Data Settings
BAR_FILE = f"{DATA_DIR}/live/NQ/bar_data.csv"
LOOKBACK = 350  # Number of historical bars to maintain

# Trading Parameters
TP_ATR_MULTIPLIER = 2.0  # Take profit ATR multiplier
SL_ATR_MULTIPLIER = 1.5  # Stop loss ATR multiplier
TICK_SIZE = 0.25  # Minimum price increment
CONTRACTS_PER_TRADE = 3  # Number of contracts to trade

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
# ENHANCED TRADING STATE
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedTradingState:
    """
    Enhanced trading state with order management and notifications.
    """
    
    def __init__(self):
        # Core components
        self.model = None
        self.px = None  # ProjectX API client
        self.market_hub = None  # SignalR market data
        self.order_manager = None  # Order lifecycle manager
        self.telegram_notifier = None  # Telegram notifications
        
        # Indicator caches
        self.f5_cache = None
        self.f15_cache = None
        self.f1h_cache = None
        
        # Timing tracking
        self.last_15m_ts = None
        self.last_1h_ts = None
        self.last_bar_timestamp = None
        
        # Data storage
        self.df_window = None
        self.bar_buffer = deque(maxlen=LOOKBACK)
        
        # Thread safety
        self.processing_lock = threading.Lock()
        
        # System state
        self.is_initialized = False
        self.reconnect_attempts = 0
        self.last_error = None
        self.startup_grace_period = True  # Prevent trading on startup
        self.first_realtime_bar_received = False
        self.user_hub = None
        
    def initialize(self):
        """Initialize all components with error recovery."""
        global CONTRACT_ID, CONTRACT_SYMBOL
        
        try:
            # Load trading model
            print("Loading model...")
            self.model = joblib.load(MODEL_FILE)
            
            # Initialize broker API
            print("Connecting to broker API...")
            self.px = ProjectXClient(
                FUTURES["topstep"]["username"], 
                FUTURES["topstep"]["api_key"]
            )
            self.px.authenticate(preferred_account_name="PRAC-V2-68606-82179008")
            
            # Find contract
            if CONTRACT_ID is None:
                print(f"Searching for contract: {CONTRACT_SEARCH}")
                contract_info = self.px.get_contract_info(CONTRACT_SEARCH)
                
                CONTRACT_ID = contract_info['id']
                CONTRACT_SYMBOL = contract_info['id']
                
                print(f"Contract found: {CONTRACT_SYMBOL}")
                print(f"  Name: {contract_info['raw']['name']}")
                print(f"  Description: {contract_info['raw']['description']}")
            
            # Initialize order manager
            print("Starting order manager...")
            self.order_manager = OrderManager(self.px)
            self.order_manager.start()

            # Initialize SignalR user hub for real-time order updates
            print("Connecting to real-time order updates...")
            self.user_hub = UserHubClient(self.px.token, self.px.account_id)

            # Set up user hub callbacks
            self.user_hub.on_order(self._on_user_order_update)
            self.user_hub.on_position(self._on_user_position_update)
            self.user_hub.on_trade(self._on_user_trade_update)
            self.user_hub.on_account(self._on_user_account_update)

            # Start user hub in separate thread
            def run_user_hub():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.user_hub.start())
                except Exception as e:
                    print(f"[ERROR] User hub error: {e}")

            threading.Thread(target=run_user_hub, daemon=True).start()
            time.sleep(1)  # Give it a moment to connect
            print("Real-time order updates: Connected")
            
            # Set up order callbacks
            self.order_manager.on_order_filled = self._on_order_filled
            self.order_manager.on_order_cancelled = self._on_order_cancelled
            self.order_manager.on_order_error = self._on_order_error
            
            # Load historical data
            print("Loading historical bars...")
            # Wait for 5-minute boundary
            NY_TZ = pytz.timezone("America/New_York")
            now = datetime.now(NY_TZ)
            secs_into_bucket = (now.minute % 5) * 60 + now.second
            wait_secs = 5*60 - secs_into_bucket
            if wait_secs > 0:
                print(f"Waiting {wait_secs:.0f}s for next 5-min boundary...")
                time.sleep(wait_secs)
                
            self.df_window = self.load_initial_bars()
            print(f"Loaded {len(self.df_window)} bars")
            
            # Convert to buffer
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
            
            # Set up market data callbacks
            self.market_hub.add_bar_handler(self.on_new_bar)
            self.market_hub.add_connection_handler(self._on_connection_change)
            
            # Start market data
            self.market_hub.start(CONTRACT_SYMBOL)
            
            # Set up Telegram notifications (optional)
            self.telegram_notifier = create_telegram_notifier()
            if self.telegram_notifier:
                setup_telegram_notifications(self.telegram_notifier, self, self.order_manager)
                print("OK Telegram notifications: Enabled")
            else:
                print("X Telegram notifications: Disabled (no config found)")
            
            self.is_initialized = True
            
        except Exception as e:
            self.last_error = e
            print(f"[FATAL] Initialization failed: {e}")
            traceback.print_exc()
            raise
    
    def load_initial_bars(self) -> pd.DataFrame:
        """Load initial historical bars from the API."""
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
    
    def check_system_health(self):
        """Check if all system components are healthy."""
        issues = []
        
        # Check market data age
        if self.last_bar_timestamp:
            if self.last_bar_timestamp.tzinfo is None:
                # If last_bar_timestamp is naive, assume it's in NY_TZ
                last_bar_tz_aware = NY_TZ.localize(self.last_bar_timestamp)
            else:
                # Convert to NY timezone
                last_bar_tz_aware = self.last_bar_timestamp.astimezone(NY_TZ)

            bar_age = (datetime.now(NY_TZ) - last_bar_tz_aware).total_seconds()
            if bar_age > 600 and check_trading_hours(datetime.now(NY_TZ)):
                print(f"last_bar_timestamp: {self.last_bar_timestamp} (tz: {self.last_bar_timestamp.tzinfo})")
                print(f"current time: {datetime.now(NY_TZ)} (tz: {datetime.now(NY_TZ).tzinfo})")
                print(f"difference: {datetime.now(NY_TZ) - self.last_bar_timestamp}")
                issues.append(f"No bars for {bar_age/60:.0f} minutes")
        
        # Check thread count
        thread_count = threading.active_count()
        if thread_count < 3:  # Main + market + user hub at minimum
            issues.append(f"Low thread count: {thread_count}")
        
        # Check order manager
        if self.order_manager and not self.order_manager._running:
            issues.append("Order manager not running")
        
        if issues:
            print(f"[HEALTH CHECK] Issues detected: {', '.join(issues)}")
            if self.telegram_notifier:
                self.telegram_notifier.send_error(f"Health check failed: {', '.join(issues)}", "System Health")
        
        return len(issues) == 0

    def on_new_bar(self, bar_data: dict):
        """Enhanced callback for new bar with error recovery."""
        with self.processing_lock:
            try:
                # ADD THIS at the beginning
                print(f"[DEBUG] on_new_bar called at {datetime.now(NY_TZ):%H:%M:%S}")
                
                if self.startup_grace_period:
                    self.startup_grace_period = False
                    self.first_realtime_bar_received = True
                    print("[INFO] First real-time bar received, trading enabled")
                # Reset reconnect counter on successful data
                self.reconnect_attempts = 0
                
                # Parse bar timestamp
                bar_time = pd.to_datetime(bar_data['t'], utc=True).tz_convert(NY_TZ)
                
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
                
                print(f"[DEBUG] Starting signal processing thread for bar {bar_time}")
                # Process signal in separate thread
                threading.Thread(
                    target=self.process_new_bar_signal,
                    args=(self.df_window.copy(), bar_time),
                    daemon=True
                ).start()

                
            except Exception as e:
                self.last_error = e
                print(f"[ERROR] Failed to process new bar: {e}")
                traceback.print_exc()
                
                # Send error notification
                if self.telegram_notifier:
                    self.telegram_notifier.send_error(str(e), "Bar Processing")
    
    def _on_user_order_update(self, *args):
        """Handle real-time order updates from SignalR with flexible arguments."""
        try:
            # Handle both (account_id, order_data) and just (order_data) formats
            if len(args) == 2:
                account_id, order_data = args
            elif len(args) == 1:
                order_data = args[0]
                account_id = self.px.account_id
            else:
                print(f"[ERROR] Unexpected args for order update: {len(args)} args")
                return
                
            order_id = order_data.get('id') or order_data.get('orderId')
            status = order_data.get('status')
            
            # Log the update
            status_name = {
                0: "Working",
                1: "Filled", 
                2: "Cancelled",
                3: "Expired",
                4: "Rejected"
            }.get(status, "Unknown")
            
            print(f"[ORDER UPDATE] Order {order_id}: {status_name}")
            
            # Update order manager's view of order status
            if hasattr(self.order_manager, '_handle_realtime_order_update'):
                self.order_manager._handle_realtime_order_update(order_id, status, order_data)
                
            # Send notification for fills
            if status == 1 and self.telegram_notifier:  # Filled
                side = "BUY" if order_data.get('side') == 0 else "SELL"
                price = order_data.get('averageFillPrice', order_data.get('price', 0))
                qty = order_data.get('filledSize', order_data.get('size', 0))
                self.telegram_notifier.send_order_fill(order_id, side, price, qty)
                
        except Exception as e:
            print(f"[ERROR] Failed to process order update: {e}")

    def _on_user_position_update(self, *args):
        """Handle real-time position updates with flexible arguments."""
        try:
            # Handle both (account_id, position_data) and just (position_data) formats
            if len(args) == 2:
                account_id, position_data = args
            elif len(args) == 1:
                position_data = args[0]
            else:
                print(f"[ERROR] Unexpected args for position update: {len(args)} args")
                return
                
            contract = position_data.get('contractName', 'Unknown')
            qty = position_data.get('netQuantity', 0)
            avg_price = position_data.get('averagePrice', 0)
            pnl = position_data.get('unrealizedPnL', 0)
            
            if qty != 0:
                print(f"[POSITION] {contract}: {qty} @ {avg_price:.2f}, PnL: ${pnl:.2f}")
        except Exception as e:
            print(f"[ERROR] Failed to process position update: {e}")
            
    def _on_user_trade_update(self, *args):
        """Handle real-time trade updates with flexible arguments."""
        try:
            # Handle both (account_id, trade_data) and just (trade_data) formats
            if len(args) == 2:
                account_id, trade_data = args
            elif len(args) == 1:
                trade_data = args[0]
            else:
                print(f"[ERROR] Unexpected args for trade update: {len(args)} args")
                return
                
            trade_id = trade_data.get('id')
            side = "BUY" if trade_data.get('side') == 0 else "SELL"
            price = trade_data.get('price', 0)
            qty = trade_data.get('quantity', 0)
            
            print(f"[TRADE] {side} {qty} @ {price:.2f}")
        except Exception as e:
            print(f"[ERROR] Failed to process trade update: {e}")
            
    def _on_user_account_update(self, *args):
        """Handle real-time account updates with flexible arguments."""
        try:
            # Account update typically just has account_data
            if len(args) >= 1:
                account_data = args[0]
            else:
                print(f"[ERROR] No data in account update")
                return
                
            balance = account_data.get('balance', 0)
            buying_power = account_data.get('buyingPower', 0)
            daily_pnl = account_data.get('dailyPnL', 0)
            
            print(f"[ACCOUNT] Balance: ${balance:.2f}, BP: ${buying_power:.2f}, Daily P&L: ${daily_pnl:.2f}")
        except Exception as e:
            print(f"[ERROR] Failed to process account update: {e}")

    def _on_connection_change(self, connected: bool):
        """Handle market data connection changes."""
        if connected:
            print(f"[{datetime.now(NY_TZ):%H:%M:%S}] Market data connected")
            self.reconnect_attempts = 0
        else:
            print(f"[{datetime.now(NY_TZ):%H:%M:%S}] Market data disconnected")
            self.reconnect_attempts += 1
    
    def _on_order_filled(self, order_id: int, order_type: str, group):
        """Handle order fill notifications."""
        print(f"[ORDER] {order_type} filled: {order_id}")
    
    def _on_order_cancelled(self, order_id: int, reason: str):
        """Handle order cancellation notifications."""
        print(f"[ORDER] Cancelled {order_id}: {reason}")
    
    def _on_order_error(self, error: Exception):
        """Handle order errors."""
        print(f"[ORDER ERROR] {error}")
        if self.telegram_notifier:
            self.telegram_notifier.send_error(str(error), "Order System")
    
    def process_new_bar_signal(self, df_window: pd.DataFrame, bar_time: datetime):
        """Enhanced signal processing with OCO order management."""
        if self.startup_grace_period:
            print("Skipping signal - waiting for first real-time bar")
            return
        
        now = datetime.now(NY_TZ)
        
        # Check trading hours
        if not check_trading_hours(now):
            print("Outside of trading hours")
            # Cancel any open orders
            self.order_manager.cancel_all_orders()
            return

        # Check for existing open orders
        try:
            active_groups = self.order_manager.get_active_groups()
            if active_groups:
                print(f"Skipping: {len(active_groups)} active order groups")
                return
        except Exception as e:
            print(f"[WARN] Failed to check active orders: {e}")
            return

        # Prepare features
        features_df = prepare_features_realtime(df_window, now, self)
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
        pred = self.model.predict(X)[0]
        
        # Map prediction to action
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

        # Send signal notification
        if self.telegram_notifier:
            self.telegram_notifier.send_trade_signal(side, price, tp_price, sl_price, bar_time)

        print(f"Entering at {price:.2f} with SL {sl_price:.2f} and TP {tp_price:.2f}")

        # Place bracket order with OCO management
        # try:
        #     order_group = place_bracket_order(
        #         self.px,
        #         self.order_manager,
        #         CONTRACT_ID,
        #         side,
        #         CONTRACTS_PER_TRADE,
        #         tp_price,
        #         sl_price
        #     )
            
        #     if order_group:
        #         print(f"[{now}] {side} bracket order placed @ {price:.2f}")
        #         print(f"  TP: {tp_price:.2f} (+{abs(tp_price - price):.2f} pts)")
        #         print(f"  SL: {sl_price:.2f} (-{abs(sl_price - price):.2f} pts)")
        #         print(f"  Order IDs: Entry={order_group.entry_order_id}, TP={order_group.tp_order_id}, SL={order_group.sl_order_id}")
        #     else:
        #         print(f"[ERROR] Failed to place bracket order")
                
        # except Exception as e:
        #     print(f"[ERROR] Order placement failed: {e}")
        #     traceback.print_exc()
            
        #     if self.telegram_notifier:
        #         self.telegram_notifier.send_error(str(e), "Order Placement")

    def check_connections_health(self):
        """Check health of all connections."""
        market_connected = self.market_hub and self.market_hub.is_connected
        user_connected = self.user_hub and self.user_hub.client and self.user_hub.client._connected
        
        if not market_connected or not user_connected:
            status = f"Market: {'✓' if market_connected else '✗'}, User: {'✓' if user_connected else '✗'}"
            print(f"[WARNING] Connection issue - {status}")
            
            if self.telegram_notifier:
                self.telegram_notifier.send_error(f"Connection issue: {status}", "SignalR")
        
        return market_connected and user_connected
    
# Create global state instance
state = EnhancedTradingState()

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (keep most of them the same)
# ═══════════════════════════════════════════════════════════════════════════

def round_tick(price: float) -> float:
    """Round price to nearest tick size."""
    if price is None:
        return None
    return round(price * 4) * 0.25

def buffer_to_dataframe(buffer: deque) -> pd.DataFrame:
    """Convert bar buffer to pandas DataFrame."""
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
    """Append new bar to CSV file for persistence."""
    try:
        bar_time = pd.to_datetime(bar_data['t'], utc=True).tz_convert(NY_TZ)
        current_time = datetime.now(NY_TZ)
        
        # Check if bar is too old (more than 30 minutes)
        bar_age = (current_time - bar_time).total_seconds()
        if bar_age > 1800:  # 30 minutes
            print(f"[WARN] Skipping stale bar from {bar_time} (age: {bar_age/60:.0f} minutes)")
            return
        
        with open(BAR_FILE, 'a') as f:
            f.write(f"{bar_time},{bar_data['o']},{bar_data['h']},{bar_data['l']},{bar_data['c']},{bar_data['v']}\n")
    except Exception as e:
        print(f"[WARN] Failed to append bar to file: {e}")

def check_trading_hours(now: datetime) -> bool:
    """Check if current time is within trading hours."""
    # Skip 4 PM - 5 PM ET (market closed)
    return not (16 <= now.hour <= 17)

# Update prepare_features_realtime to accept state parameter
def prepare_features_realtime(df_window: pd.DataFrame, now: datetime, trading_state) -> pd.DataFrame:
    """Prepare feature set for model prediction using trading state."""
    # Always compute 5min indicators (base timeframe)
    trading_state.f5_cache = compute_indicators_cached(df_window, '5min', ['all'], trading_state)
    
    # Compute 15min indicators when needed
    if is_new_timeframe_bar(now, trading_state.last_15m_ts, 15):
        df15 = df_window.resample('15min', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        trading_state.f15_cache = compute_all_indicators(
            df15.copy(), 
            suffix='_15min', 
            features=['volume_trend', 'prev_swing', 'trend', 'poc', 'adx', 'ema', 'atr']
        )
        trading_state.last_15m_ts = now
    
    # Compute 1h indicators when needed
    if is_new_timeframe_bar(now, trading_state.last_1h_ts, 60):
        df1h = df_window.resample('1h', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        trading_state.f1h_cache = compute_all_indicators(
            df1h.copy(), 
            suffix='_1h', 
            features=['adx', 'poc']
        )
        trading_state.last_1h_ts = now
    
    # Merge all timeframes into final feature set
    if trading_state.f15_cache is not None and trading_state.f1h_cache is not None:
        # Get latest 5min row
        f5_last = trading_state.f5_cache.iloc[[-1]].copy()
        
        # Get latest values from higher timeframes
        f15_latest = trading_state.f15_cache.filter(regex='_15min').iloc[[-1]]
        f1h_latest = trading_state.f1h_cache.filter(regex='_1h').iloc[[-1]]
        
        # Merge all features
        for col in f15_latest.columns:
            f5_last[col] = f15_latest[col].iloc[0]
        for col in f1h_latest.columns:
            f5_last[col] = f1h_latest[col].iloc[0]
            
        return f5_last
    
    # Return whatever we have if higher timeframes not ready
    return trading_state.f5_cache.tail(1) if trading_state.f5_cache is not None else pd.DataFrame()

def compute_indicators_cached(df: pd.DataFrame, timeframe: str, features: list, trading_state) -> pd.DataFrame:
    """Compute technical indicators with intelligent caching."""
    if timeframe == '5min':
        # For 5min, try incremental computation
        if trading_state.f5_cache is not None and len(df) > len(trading_state.f5_cache):
            new_rows = len(df) - len(trading_state.f5_cache)
            if new_rows <= 5:  # Only if a few new bars
                # Compute indicators for recent data with context
                new_data = df.tail(50)  # Get enough context for indicators
                new_indicators = compute_all_indicators(new_data, suffix='_5min', features=features)
                new_indicators = session_times(new_indicators)
                # Append only new rows to cache
                result = pd.concat([trading_state.f5_cache, new_indicators.tail(new_rows)])
                return result.tail(LOOKBACK)
    
    # Full computation for other timeframes or when cache is stale
    result = compute_all_indicators(df.copy(), suffix=f'_{timeframe}', features=features)
    if timeframe == '5min':
        result = session_times(result)
    return result

def is_new_timeframe_bar(now: datetime, last_ts: datetime, minutes: int) -> bool:
    """Check if we need to compute new indicators for a timeframe."""
    if last_ts is None:
        return True
    return (now - last_ts).total_seconds() >= minutes * 60

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Enhanced main execution loop with better error handling."""
    print("=" * 70)
    print("ENHANCED LIVE TRADING SYSTEM")
    print("=" * 70)
    
    try:
        # Initialize everything
        state.initialize()
        print(f"\nOK System initialized successfully")
        print(f"OK Contract: {CONTRACT_SYMBOL}")
        print(f"OK Historical bars: {len(state.df_window)}")
        print(f"OK Real-time data: Connected")
        print(f"OK Order manager: Running")
        print("\nWaiting for market data...\n")

        from trading_monitor_v2 import TradingMonitor
        monitor = TradingMonitor(
            state.px,
            state.px.account_id,
            order_manager=state.order_manager
        )
        threading.Thread(target=monitor.run, daemon=True).start()
        
        # Show initial market quote
        time.sleep(2)
        latest_quote = state.market_hub.get_latest_quote(CONTRACT_SYMBOL)
        if latest_quote:
            bid = latest_quote.get('bestBid', 'N/A')
            ask = latest_quote.get('bestAsk', 'N/A')
            print(f"Current market: Bid={bid} Ask={ask}")
        
    except Exception as e:
        print(f"\n[FATAL] Initialization failed: {e}")
        traceback.print_exc()
        return

    # Main loop with health monitoring
    try:
        print("\nSystem running. Press Ctrl+C to stop.\n")
        
        last_heartbeat = time.time()
        heartbeat_interval = 60  # seconds
        last_data_received = time.time()  # ADD THIS
        
        while True:
            time.sleep(5)
            
            # Periodic health check
            current_time = time.time()
            
            # ADD THIS: Update last_data_received if we have recent bar
            if state.last_bar_timestamp:
                time_since_last_bar = (datetime.now(NY_TZ) - state.last_bar_timestamp).total_seconds()
                if time_since_last_bar < 360:  # Less than 6 minutes old
                    last_data_received = current_time
            
            # Send heartbeat
            if current_time - last_heartbeat >= heartbeat_interval:
                if state.market_hub and state.market_hub.is_connected:
                    quote = state.market_hub.get_latest_quote(CONTRACT_SYMBOL)
                    if quote:
                        bid = quote.get('bestBid', 'N/A')
                        ask = quote.get('bestAsk', 'N/A')
                        spread = float(ask) - float(bid) if bid != 'N/A' and ask != 'N/A' else 'N/A'
                        
                        # ADD THIS: Check if connection is stale
                        time_since_data = current_time - last_data_received
                        if time_since_data > 600:  # 10 minutes without data
                            print(f"[{datetime.now(NY_TZ):%H:%M:%S}] WARNING: Connection stale - no data for {time_since_data:.0f}s, forcing reconnect...")
                            # Force reconnection by stopping and restarting market hub
                            try:
                                state.market_hub.stop()
                                time.sleep(2)
                                state.market_hub.start(CONTRACT_SYMBOL)
                                print("Market hub restarted")
                            except Exception as e:
                                print(f"Failed to restart market hub: {e}")
                        else:
                            heartbeat_msg = f"[{datetime.now(NY_TZ):%H:%M:%S}] Heartbeat: Connected | Bid={bid} Ask={ask} Spread={spread}"
                            print(heartbeat_msg)
                        
                        # Send market update to Telegram
                        if state.telegram_notifier and bid != 'N/A' and ask != 'N/A':
                            try:
                                state.telegram_notifier.send_market_update(
                                    float(bid), float(ask), float(quote.get('lastPrice', 0))
                                )
                            except ValueError:
                                pass
                else:
                    print(f"[{datetime.now(NY_TZ):%H:%M:%S}] Heartbeat: Disconnected - attempting reconnection...")
                
                last_heartbeat = current_time
            
            # Check for stale data (no bar in 10 minutes during market hours)
            if state.last_bar_timestamp and state.first_realtime_bar_received and check_trading_hours(datetime.now(NY_TZ)):
                time_since_last_bar = (datetime.now(NY_TZ) - state.last_bar_timestamp).total_seconds()
                if time_since_last_bar > 600:  # 10 minutes
                    print(f"[WARNING] No bar received for {time_since_last_bar:.0f} seconds")
                    
                    # Only send one notification per disconnect, not every 5 seconds
                    if time_since_last_bar < 615 and state.telegram_notifier:
                        state.telegram_notifier.send_error(
                            f"No market data for {time_since_last_bar/60:.0f} minutes",
                            "Data Feed"
                        )

            if current_time % 300 < 5:  # Every 5 minutes
                state.check_system_health()
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        
        # Graceful shutdown
        if state.order_manager:
            print("- Stopping order manager...")
            state.order_manager.stop()

        if state.user_hub:
            print("- Disconnecting user hub...")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(state.user_hub.stop())
            except:
                pass
            
        if state.market_hub:
            print("- Disconnecting market data...")
            state.market_hub.stop()
            
        if state.telegram_notifier:
            print("- Stopping Telegram notifications...")
            state.telegram_notifier.stop()
            
        print("- Shutdown complete")
        
    except Exception as e:
        print(f"\n[FATAL] Unexpected error: {e}")
        traceback.print_exc()
        
        # Emergency shutdown
        if state.order_manager:
            state.order_manager.stop()
        if state.market_hub:
            state.market_hub.stop()
        if state.telegram_notifier:
            state.telegram_notifier.stop()


if __name__ == "__main__":
    main()