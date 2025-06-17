"""
TopStepX SignalR Market Data Client

This module provides real-time market data streaming via SignalR WebSockets.
It aggregates tick data into 5-minute OHLCV bars and provides callbacks for
trade signals.

Features:
- Real-time quote, trade, and market depth streaming
- Automatic tick-to-bar aggregation
- Thread-safe data handling
- Automatic reconnection on disconnect
- Efficient bar completion detection

Author: Trading Bot
Date: 2024
"""

import json
import logging
import queue
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from threading import Thread, Event, Lock
from typing import Dict, Optional, Callable, List

import pandas as pd
from signalrcore.hub_connection_builder import HubConnectionBuilder

# Configure logging
logger = logging.getLogger(__name__)


class BarAggregator:
    """
    Aggregates tick/trade data into time-based OHLCV bars.
    
    This class maintains running bars for each contract and emits
    completed bars via callbacks when a new bar period starts.
    """
    
    def __init__(self, bar_interval_minutes: int = 5):
        """
        Initialize the bar aggregator.
        
        Args:
            bar_interval_minutes: Bar interval in minutes (default: 5)
        """
        self.bar_interval = bar_interval_minutes
        self.current_bars = {}  # contract_id -> {bar_time -> bar_data}
        self.lock = Lock()
        
    def process_trade(self, contract_id: str, trade_data: dict, callback: Callable):
        """
        Process a trade tick and potentially emit a completed bar.
        
        Args:
            contract_id: Contract identifier
            trade_data: Trade data with price, volume, timestamp
            callback: Function to call when a bar is completed
        """
        with self.lock:
            # Parse trade data
            price = float(trade_data.get('price', 0))
            volume = int(trade_data.get('volume', 0))
            timestamp = pd.to_datetime(trade_data.get('timestamp', datetime.now(timezone.utc)))
            
            # Calculate bar boundary
            bar_time = self._get_bar_time(timestamp)
            
            # Initialize contract bars if needed
            if contract_id not in self.current_bars:
                self.current_bars[contract_id] = {}
                
            bars = self.current_bars[contract_id]
            
            # Check if this trade belongs to a new bar period
            if bar_time not in bars:
                # Emit previous bar if it exists and is complete
                if bars:
                    prev_bar_time = max(bars.keys())
                    if bar_time > prev_bar_time:
                        completed_bar = bars[prev_bar_time]
                        completed_bar['t'] = prev_bar_time.isoformat()
                        completed_bar['contractId'] = contract_id
                        callback(completed_bar)
                        # Remove the completed bar
                        del bars[prev_bar_time]
                
                # Start new bar
                bars[bar_time] = {
                    'o': price,
                    'h': price,
                    'l': price,
                    'c': price,
                    'v': volume
                }
            else:
                # Update existing bar
                bar = bars[bar_time]
                bar['h'] = max(bar['h'], price)
                bar['l'] = min(bar['l'], price)
                bar['c'] = price
                bar['v'] += volume
                
    def _get_bar_time(self, timestamp: datetime) -> datetime:
        """
        Round timestamp down to nearest bar interval.
        
        Args:
            timestamp: Raw timestamp
            
        Returns:
            Rounded bar timestamp
        """
        minutes = (timestamp.minute // self.bar_interval) * self.bar_interval
        return timestamp.replace(minute=minutes, second=0, microsecond=0)
        
    def flush_completed_bars(self, callback: Callable):
        """
        Emit any bars that are complete based on current time.
        
        This method is called periodically to ensure bars are emitted
        even during low-activity periods.
        
        Args:
            callback: Function to call for each completed bar
        """
        with self.lock:
            current_time = datetime.now(timezone.utc)
            current_bar_time = self._get_bar_time(current_time)
            
            for contract_id, bars in self.current_bars.items():
                completed_times = [t for t in bars.keys() if t < current_bar_time]
                for bar_time in completed_times:
                    completed_bar = bars[bar_time]
                    completed_bar['t'] = bar_time.isoformat()
                    completed_bar['contractId'] = contract_id
                    callback(completed_bar)
                    del bars[bar_time]


class TopStepMarketHub:
    """
    TopStepX SignalR market data client for real-time streaming.
    
    This class manages the SignalR WebSocket connection to TopStepX's
    real-time data feed, handling quotes, trades, and market depth.
    """
    
    def __init__(self, token: str):
        """
        Initialize the market hub.
        
        Args:
            token: JWT authentication token from TopStepX
        """
        self.token = token
        self.hub_url = f"https://rtc.topstepx.com/hubs/market?access_token={token}"
        self.hub_connection = None
        self.is_connected = False
        self.connection_event = Event()
        
        # Initialize components
        self.bar_aggregator = BarAggregator(bar_interval_minutes=5)
        
        # Callback lists
        self.bar_callbacks: List[Callable] = []
        self.quote_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        
        # Thread-safe queue for completed bars
        self.bar_queue = queue.Queue()
        
        # Track subscribed contracts
        self.subscribed_contracts = set()
        
        # Start background timer for bar completion checks
        self._start_bar_timer()
        
    def _start_bar_timer(self):
        """
        Start a background thread to periodically check for completed bars.
        
        This ensures bars are emitted even during periods of low trading activity.
        """
        def timer_loop():
            while True:
                Event().wait(10)  # Check every 10 seconds
                if self.is_connected:
                    self.bar_aggregator.flush_completed_bars(self._on_bar_completed)
                    
        timer_thread = Thread(target=timer_loop, daemon=True)
        timer_thread.start()
        
    def build_connection(self):
        """
        Build the SignalR connection with TopStepX configuration.
        
        Uses WebSocket transport with automatic reconnection.
        """
        self.hub_connection = HubConnectionBuilder()\
            .with_url(self.hub_url, options={
                "skip_negotiation": True,
                "transport": 1,  # WebSockets only
                "access_token_factory": lambda: self.token,
                "timeout": 10000
            })\
            .with_automatic_reconnect({
                "type": "intervals",
                "intervals": [0, 2, 5, 10, 30]  # Reconnect intervals in seconds
            })\
            .build()
        
        # Register connection event handlers
        self.hub_connection.on_open(self._on_open)
        self.hub_connection.on_close(self._on_close)
        self.hub_connection.on_error(self._on_error)
        self.hub_connection.on_reconnected(self._on_reconnected)
        
        # Register TopStepX market data handlers
        self.hub_connection.on("GatewayQuote", self._on_quote_received)
        self.hub_connection.on("GatewayTrade", self._on_trade_received)
        self.hub_connection.on("GatewayDepth", self._on_depth_received)
        
    # Connection Event Handlers
    
    def _on_open(self):
        """Handle successful connection."""
        logger.info("SignalR connection established to TopStepX")
        self.is_connected = True
        self.connection_event.set()
        
    def _on_close(self):
        """Handle connection closed."""
        logger.warning("SignalR connection closed")
        self.is_connected = False
        self.connection_event.clear()
        
    def _on_error(self, data):
        """Handle connection error."""
        logger.error(f"SignalR error: {data}")
        
    def _on_reconnected(self, connection_id):
        """Handle successful reconnection - resubscribe to all contracts."""
        logger.info(f"RTC Connection Reconnected: {connection_id}")
        for contract_id in self.subscribed_contracts:
            self._subscribe_contract(contract_id)
    
    # Market Data Event Handlers
    
    def _on_quote_received(self, contract_id: str, data: dict):
        """
        Handle quote (bid/ask) data from TopStepX.
        
        Args:
            contract_id: Contract identifier
            data: Quote data containing bid/ask prices and sizes
        """
        try:
            # Notify quote callbacks
            for callback in self.quote_callbacks:
                try:
                    callback(contract_id, data)
                except Exception as e:
                    logger.error(f"Quote callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing quote data: {e}")
            
    def _on_trade_received(self, contract_id: str, data: dict):
        """
        Handle trade data and aggregate into bars.
        
        Args:
            contract_id: Contract identifier
            data: Trade data containing price, volume, timestamp
        """
        try:
            # Process trade for bar aggregation
            self.bar_aggregator.process_trade(contract_id, data, self._on_bar_completed)
            
            # Notify trade callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(contract_id, data)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
            
    def _on_depth_received(self, contract_id: str, data: dict):
        """
        Handle market depth (order book) data.
        
        Args:
            contract_id: Contract identifier
            data: Market depth data
        """
        # Available for future use if needed
        pass
        
    def _on_bar_completed(self, bar_data: dict):
        """
        Handle a completed bar from the aggregator.
        
        Args:
            bar_data: Completed OHLCV bar data
        """
        # Add to queue for thread-safe processing
        self.bar_queue.put(bar_data)
        
        # Notify bar callbacks
        for callback in self.bar_callbacks:
            try:
                callback(bar_data)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")
    
    # Public Methods
    
    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to the SignalR hub.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            self.build_connection()
            self.hub_connection.start()
            
            # Wait for connection
            if self.connection_event.wait(timeout):
                return True
            else:
                logger.error("Connection timeout")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the SignalR hub."""
        if self.hub_connection:
            self.hub_connection.stop()
            self.is_connected = False
            
    def subscribe_contract(self, contract_id: str):
        """
        Subscribe to all market data for a contract.
        
        Args:
            contract_id: Contract identifier (e.g., 'CON.F.US.NQ.H25')
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to market hub")
            
        self.subscribed_contracts.add(contract_id)
        self._subscribe_contract(contract_id)
        
    def _subscribe_contract(self, contract_id: str):
        """
        Internal method to subscribe to contract data.
        
        Args:
            contract_id: Contract identifier
        """
        try:
            # Subscribe to quotes, trades, and depth as per TopStepX API
            self.hub_connection.send("SubscribeContractQuotes", [contract_id])
            self.hub_connection.send("SubscribeContractTrades", [contract_id])
            self.hub_connection.send("SubscribeContractMarketDepth", [contract_id])
            logger.info(f"Subscribed to market data for contract {contract_id}")
        except Exception as e:
            logger.error(f"Failed to subscribe contract {contract_id}: {e}")
            
    def unsubscribe_contract(self, contract_id: str):
        """
        Unsubscribe from contract market data.
        
        Args:
            contract_id: Contract identifier
        """
        if not self.is_connected:
            return
            
        try:
            self.hub_connection.send("UnsubscribeContractQuotes", [contract_id])
            self.hub_connection.send("UnsubscribeContractTrades", [contract_id])
            self.hub_connection.send("UnsubscribeContractMarketDepth", [contract_id])
            self.subscribed_contracts.discard(contract_id)
        except Exception as e:
            logger.error(f"Failed to unsubscribe contract {contract_id}: {e}")
            
    def add_bar_callback(self, callback: Callable):
        """Add a callback for completed bar notifications."""
        self.bar_callbacks.append(callback)
        
    def add_quote_callback(self, callback: Callable):
        """Add a callback for quote updates."""
        self.quote_callbacks.append(callback)
        
    def add_trade_callback(self, callback: Callable):
        """Add a callback for trade updates."""
        self.trade_callbacks.append(callback)
        
    def get_latest_bar(self, timeout: float = 0.1) -> Optional[Dict]:
        """
        Get the latest bar from the queue (non-blocking).
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Bar data dict or None if no bar available
        """
        try:
            return self.bar_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_all_pending_bars(self) -> List[Dict]:
        """
        Get all pending bars from the queue.
        
        Returns:
            List of bar data dicts
        """
        bars = []
        while True:
            try:
                bar = self.bar_queue.get_nowait()
                bars.append(bar)
            except queue.Empty:
                break
        return bars


class TopStepMarketDataManager:
    """
    High-level manager for TopStepX SignalR market data.
    
    This class provides a simplified interface for integrating
    real-time market data into trading strategies.
    """
    
    def __init__(self, token: str):
        """
        Initialize the market data manager.
        
        Args:
            token: JWT authentication token from TopStepX
        """
        self.market_hub = TopStepMarketHub(token)
        self.latest_bars = {}
        self.latest_quotes = {}
        self.bar_callbacks = []
        
    def start(self, contract_id: str):
        """
        Start market data streaming for a contract.
        
        Args:
            contract_id: Contract identifier (e.g., 'CON.F.US.NQ.H25')
        """
        # Connect to hub
        if not self.market_hub.connect():
            raise RuntimeError("Failed to connect to TopStepX market hub")
            
        # Subscribe to contract
        self.market_hub.subscribe_contract(contract_id)
        
        # Add internal callbacks
        self.market_hub.add_bar_callback(self._store_bar)
        self.market_hub.add_quote_callback(self._store_quote)
        
        logger.info(f"Market data streaming started for {contract_id}")
        
    def stop(self):
        """Stop market data streaming."""
        self.market_hub.disconnect()
        
    def _store_bar(self, bar_data: Dict):
        """
        Store latest bar for each contract.
        
        Args:
            bar_data: Completed bar data
        """
        contract_id = bar_data.get('contractId')
        if contract_id:
            self.latest_bars[contract_id] = bar_data
            
            # Notify external callbacks
            for callback in self.bar_callbacks:
                try:
                    callback(bar_data)
                except Exception as e:
                    logger.error(f"External bar callback error: {e}")
                    
    def _store_quote(self, contract_id: str, quote_data: Dict):
        """
        Store latest quote for each contract.
        
        Args:
            contract_id: Contract identifier
            quote_data: Quote data
        """
        self.latest_quotes[contract_id] = quote_data
                    
    def add_bar_handler(self, callback: Callable):
        """Add an external bar completion handler."""
        self.bar_callbacks.append(callback)
        
    def get_latest_bar(self, contract_id: str) -> Optional[Dict]:
        """
        Get latest completed bar for a contract.
        
        Args:
            contract_id: Contract identifier
            
        Returns:
            Bar data dict or None
        """
        return self.latest_bars.get(contract_id)
        
    def get_latest_quote(self, contract_id: str) -> Optional[Dict]:
        """
        Get latest quote for a contract.
        
        Args:
            contract_id: Contract identifier
            
        Returns:
            Quote data dict or None
        """
        return self.latest_quotes.get(contract_id)
        
    def wait_for_new_bar(self, contract_id: str, timeout: float = 310) -> Optional[Dict]:
        """
        Wait for the next bar to complete.
        
        Args:
            contract_id: Contract identifier
            timeout: Maximum time to wait in seconds (default: 5min + 10sec)
            
        Returns:
            New bar data dict or None if timeout
        """
        start_time = datetime.now()
        current_bar = self.latest_bars.get(contract_id)
        current_timestamp = current_bar.get('t') if current_bar else None
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            latest = self.latest_bars.get(contract_id)
            if latest and latest.get('t') != current_timestamp:
                return latest
            Event().wait(0.1)
            
        return None