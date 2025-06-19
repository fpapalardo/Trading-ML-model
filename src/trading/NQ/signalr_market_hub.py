"""
Enhanced SignalR WebSocket Client with Auto-Reconnect

This enhanced version adds automatic reconnection capability with exponential backoff
and maintains subscription state across reconnections.
"""

import asyncio
import json
import logging
from typing import Dict, Optional, Callable, List, Set
from datetime import datetime
from collections import defaultdict, deque
from threading import Lock
import websockets
from urllib.parse import quote
import pandas as pd
from datetime import timezone, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReconnectingSignalRClient:
    """SignalR WebSocket client with automatic reconnection."""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = "wss://rtc.topstepx.com/hubs/market"
        self.ws = None
        self.running = False
        self.callbacks = {
            'GatewayQuote': [],
            'GatewayTrade': [],
            'GatewayDepth': []
        }
        self.message_id = 0
        self._handshake_ack = asyncio.Event()
        
        # Reconnection settings
        self.reconnect_interval = 5  # Initial reconnect delay in seconds
        self.max_reconnect_interval = 60  # Maximum reconnect delay
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = None  # None = infinite attempts
        
        # Subscription tracking for reconnection
        self.subscribed_contracts: Set[str] = set()
        self._reconnecting = False
        self._connection_lock = asyncio.Lock()
        
    async def connect(self) -> bool:
        """Connect to TopStepX SignalR hub with reconnection logic."""
        async with self._connection_lock:
            if self.ws and not self.ws.closed:
                logger.info("Already connected")
                return True
                
            try:
                # Build WebSocket URL
                ws_url = f"{self.base_url}?access_token={self.token}"
                logger.info("Connecting to TopStepX SignalR hub...")

                # Connect with relaxed ping settings
                self.ws = await websockets.connect(
                    ws_url,
                    ping_interval=60,
                    ping_timeout=30,
                    close_timeout=10
                )
                self.running = True
                self.reconnect_attempts = 0  # Reset on successful connection
                logger.info("WebSocket connected successfully")

                # Clear any old handshake event
                self._handshake_ack.clear()

                # Send SignalR handshake
                await self._send_handshake()

                # Start processing incoming messages
                asyncio.create_task(self._message_handler())

                # Wait for handshake
                await asyncio.wait_for(self._handshake_ack.wait(), timeout=3)
                logger.info("Handshake confirmed—hub is ready")

                # Re-subscribe to any previously subscribed contracts
                if self._reconnecting and self.subscribed_contracts:
                    logger.info(f"Re-subscribing to {len(self.subscribed_contracts)} contracts")
                    for contract_id in self.subscribed_contracts:
                        await self._subscribe_contract_internal(contract_id)
                
                self._reconnecting = False
                return True
                
            except asyncio.TimeoutError:
                logger.error("Handshake timeout—server never confirmed")
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                
            self.running = False
            return False
    
    async def _reconnect_loop(self):
        """Automatic reconnection with exponential backoff."""
        self._reconnecting = True
        
        while not self.running:
            # Check if we've exceeded max attempts
            if (self.max_reconnect_attempts is not None and 
                self.reconnect_attempts >= self.max_reconnect_attempts):
                logger.error("Maximum reconnection attempts reached")
                self.running = False
                break
            
            # Calculate backoff delay
            delay = min(
                self.reconnect_interval * (2 ** self.reconnect_attempts),
                self.max_reconnect_interval
            )
            
            logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts + 1})")
            await asyncio.sleep(delay)
            
            self.reconnect_attempts += 1
            
            # Try to reconnect
            if await self.connect():
                logger.info("Reconnection successful")
                break
    
    async def _send_handshake(self):
        """Send SignalR handshake message."""
        handshake = {
            "protocol": "json",
            "version": 1
        }
        message = json.dumps(handshake) + "\x1e"
        await self.ws.send(message)
        logger.debug(f"Sent handshake")
    
    async def _message_handler(self):
        """Handle incoming messages from the WebSocket."""
        try:
            while self.running and self.ws:
                message = await self.ws.recv()
                
                # SignalR messages are terminated with \x1e
                messages = message.split('\x1e')
                
                for msg in messages:
                    if msg:
                        await self._process_message(msg)
                        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self.running = False
            
            # Start reconnection
            if not self._reconnecting:
                asyncio.create_task(self._reconnect_loop())
                
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            self.running = False
            
            # Start reconnection
            if not self._reconnecting:
                asyncio.create_task(self._reconnect_loop())
    
    async def _process_message(self, message: str):
        """Process a SignalR message."""
        try:
            # Handle empty handshake response
            if message == "{}":
                logger.info("Handshake acknowledged")
                self._handshake_ack.set()
                return
                
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 1:  # Invocation
                target = data.get('target')
                arguments = data.get('arguments', [])
                
                logger.debug(f"Received {target} with {len(arguments)} arguments")
                
                # Call registered callbacks
                if target in self.callbacks:
                    for callback in self.callbacks[target]:
                        try:
                            await callback(*arguments)
                        except Exception as e:
                            logger.error(f"Callback error for {target}: {e}", exc_info=True)
                            
            elif msg_type == 6:  # Ping
                # Respond with pong immediately
                pong = {"type": 6}
                await self.ws.send(json.dumps(pong) + "\x1e")
                logger.debug("Received ping, sent pong")
                
            elif msg_type == 7:  # Close
                error = data.get('error')
                if error:
                    logger.error(f"Server error: {error}")
                    self.running = False
                else:
                    logger.debug(f"Received close message: {data}")
                    
            elif msg_type == 3:  # Result
                logger.debug(f"Invocation result: {data}")
                if 'error' in data:
                    logger.error(f"Invocation error: {data['error']}")
                    
            else:
                logger.debug(f"Received message type {msg_type}: {data}")
                
        except json.JSONDecodeError:
            if message:
                logger.error(f"Failed to decode message: {message[:100]}...")
        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)
    
    def _get_next_id(self):
        """Get next invocation ID."""
        self.message_id += 1
        return str(self.message_id)
    
    async def invoke(self, method: str, *args):
        """Invoke a SignalR hub method."""
        if not self.ws or not self.running:
            raise RuntimeError("Not connected")
        
        # SignalR invocation message format
        message = {
            "type": 1,  # Invocation
            "invocationId": self._get_next_id(),
            "target": method,
            "arguments": list(args)
        }
        
        msg_str = json.dumps(message) + "\x1e"
        logger.debug(f"Invoking {method} with args: {args}")
        await self.ws.send(msg_str)
    
    async def _subscribe_contract_internal(self, contract_id: str):
        """Internal method to subscribe to a contract."""
        await self.invoke("SubscribeContractQuotes", contract_id)
        await self.invoke("SubscribeContractTrades", contract_id)
        await self.invoke("SubscribeContractMarketDepth", contract_id)
    
    async def subscribe_contract(self, contract_id: str):
        """Subscribe to market data for a contract."""
        if not self.running:
            logger.warning("Cannot subscribe - not connected")
            return
            
        logger.info(f"Subscribing to contract: {contract_id}")
        try:
            # Track subscription for reconnection
            self.subscribed_contracts.add(contract_id)
            await self._subscribe_contract_internal(contract_id)
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            raise
    
    async def unsubscribe_contract(self, contract_id: str):
        """Unsubscribe from market data for a contract."""
        # Remove from tracked subscriptions
        self.subscribed_contracts.discard(contract_id)
        
        await self.invoke("UnsubscribeContractQuotes", contract_id)
        await self.invoke("UnsubscribeContractTrades", contract_id)
        await self.invoke("UnsubscribeContractMarketDepth", contract_id)
    
    def on(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    async def disconnect(self):
        """Disconnect from the WebSocket."""
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None


# Update TopStepXSignalRClient to inherit from ReconnectingSignalRClient
class TopStepXSignalRClient(ReconnectingSignalRClient):
    """TopStepX SignalR client with reconnection support."""
    pass


# Enhanced Market Hub with better error handling
class TopStepMarketHub:
    """TopStepX SignalR market data client with enhanced reliability."""
    
    def __init__(self, token: str):
        self.client = TopStepXSignalRClient(token)
        self.bar_aggregator = BarAggregator(bar_interval_minutes=5)
        self.bar_callbacks = []
        self.quote_callbacks = []
        self.trade_callbacks = []
        self.connection_callbacks = []  # New: connection status callbacks
        self.latest_bars = {}
        self.latest_quotes = {}
        self.subscribed_contracts = set()
        self._connected = False
        
    async def connect(self) -> bool:
        """Connect to the market hub."""
        # Register internal handlers
        self.client.on("GatewayQuote", self._on_quote_received)
        self.client.on("GatewayTrade", self._on_trade_received)
        self.client.on("GatewayDepth", self._on_depth_received)
        
        # Connect
        success = await self.client.connect()
        
        if success:
            self._connected = True
            # Start bar flush timer
            asyncio.create_task(self._bar_timer())
            
            # Notify connection callbacks
            for callback in self.connection_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(True)
                    else:
                        callback(True)
                except Exception as e:
                    logger.error(f"Connection callback error: {e}")
        else:
            self._connected = False
            
        return success
    
    async def _bar_timer(self):
        """Periodically check for completed bars and connection health."""
        last_connection_state = self._connected
        
        while self.client.running:
            await asyncio.sleep(10)
            self.bar_aggregator.flush_completed_bars(self._on_bar_completed)
            
            # Check connection state changes
            current_state = self.client.ws and self.client.ws.state.name == 'OPEN'
            if current_state != last_connection_state:
                self._connected = current_state
                last_connection_state = current_state
                
                # Notify connection callbacks
                for callback in self.connection_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(current_state)
                        else:
                            callback(current_state)
                    except Exception as e:
                        logger.error(f"Connection callback error: {e}")
    
    def add_connection_callback(self, callback: Callable):
        """Add a connection status callback."""
        self.connection_callbacks.append(callback)
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected and self.client.running
    
    # ... rest of the methods remain the same ...
    
    async def _on_quote_received(self, contract_id: str, data):
        """Handle quote data."""
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                data = data[0]
            else:
                logger.warning(f"Unexpected quote format: {data}")
                return
        
        self.latest_quotes[contract_id] = data
        
        if 'lastPrice' in data and data['lastPrice'] is not None:
            synthetic_trade = {
                'price': data['lastPrice'],
                'volume': 0,
                'timestamp': data.get('timestamp', datetime.now(timezone.utc).isoformat())
            }
            logger.debug(f"Processing synthetic trade from quote: price={synthetic_trade['price']}")
            self.bar_aggregator.process_trade(contract_id, synthetic_trade, self._on_bar_completed)
        
        for callback in self.quote_callbacks:
            try:
                await callback(contract_id, data)
            except Exception as e:
                logger.error(f"Quote callback error: {e}")
    
    async def _on_trade_received(self, contract_id: str, data: dict):
        """Handle trade data."""
        self.bar_aggregator.process_trade(contract_id, data, self._on_bar_completed)
        
        for callback in self.trade_callbacks:
            try:
                await callback(contract_id, data)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
    
    async def _on_depth_received(self, contract_id: str, data: dict):
        """Handle market depth data."""
        pass
    
    def _on_bar_completed(self, bar_data: dict):
        """Handle completed bar."""
        contract_id = bar_data.get('contractId')
        if contract_id:
            self.latest_bars[contract_id] = bar_data
            
        for callback in self.bar_callbacks:
            try:
                callback(bar_data)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")
    
    async def subscribe_contract(self, contract_id: str):
        """Subscribe to contract data."""
        self.subscribed_contracts.add(contract_id)
        await self.client.subscribe_contract(contract_id)
    
    async def disconnect(self):
        """Disconnect from the hub."""
        await self.client.disconnect()
    
    def add_bar_callback(self, callback: Callable):
        """Add a bar completion callback."""
        self.bar_callbacks.append(callback)
    
    def add_quote_callback(self, callback: Callable):
        """Add a quote callback."""
        self.quote_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """Add a trade callback."""
        self.trade_callbacks.append(callback)
    
    def get_latest_bar(self, contract_id: str) -> Optional[Dict]:
        """Get latest bar for a contract."""
        return self.latest_bars.get(contract_id)
    
    def get_latest_quote(self, contract_id: str) -> Optional[Dict]:
        """Get latest quote for a contract."""
        return self.latest_quotes.get(contract_id)


# Keep BarAggregator class unchanged
class BarAggregator:
    """Aggregates tick/trade data into time-based OHLCV bars."""
    
    def __init__(self, bar_interval_minutes: int = 5):
        self.bar_interval = bar_interval_minutes
        self.current_bars = {}  # contract_id -> {bar_time -> bar_data}
        self.lock = Lock()
        
    def process_trade(self, contract_id: str, trade_data: dict, callback: Callable):
        """Process a trade tick and potentially emit a completed bar."""
        if isinstance(trade_data, list):
            for single_trade in trade_data:
                self.process_trade(contract_id, single_trade, callback)
            return
    
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
                # Emit previous bar if it exists
                if bars:
                    prev_bar_time = max(bars.keys())
                    if bar_time > prev_bar_time:
                        # Check if the bar is not too old
                        current_time = datetime.now(timezone.utc)
                        bar_age = (current_time - prev_bar_time).total_seconds()
                        
                        # Only emit bars that are less than 30 minutes old
                        if bar_age < 1800:  # 30 minutes
                            completed_bar = bars[prev_bar_time]
                            completed_bar['t'] = prev_bar_time.isoformat()
                            completed_bar['contractId'] = contract_id
                            callback(completed_bar)
                        else:
                            logger.debug(f"Skipping stale bar from {prev_bar_time}")
                        
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
        """Round timestamp down to nearest bar interval."""
        minutes = (timestamp.minute // self.bar_interval) * self.bar_interval
        return timestamp.replace(minute=minutes, second=0, microsecond=0)
        
    def flush_completed_bars(self, callback: Callable):
        """Emit any bars that are complete based on current time."""
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

# Update the TopStepMarketDataManager class:

class TopStepMarketDataManager:
    """Synchronous wrapper for the async SignalR client."""
    
    def __init__(self, token: str):
        import threading
        self.token = token
        self.market_hub = None
        self.loop = None
        self.thread = None
        self._running = False
        self._connected_event = threading.Event()
        self._pending_callbacks = {
            'bar': [],
            'quote': [],
            'trade': [],
            'connection': []
        }
        
    def start(self, contract_id: str):
        """Start market data streaming."""
        import threading
        
        def run_async_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            async def async_main():
                try:
                    self.market_hub = TopStepMarketHub(self.token)
                    
                    # Add any pending callbacks
                    for callback in self._pending_callbacks['bar']:
                        self.market_hub.add_bar_callback(callback)
                    for callback in self._pending_callbacks['quote']:
                        self.market_hub.add_quote_callback(callback)
                    for callback in self._pending_callbacks['trade']:
                        self.market_hub.add_trade_callback(callback)
                    
                    for cb in self._pending_callbacks['connection']:
                        self.market_hub.add_connection_callback(cb)
                    # Connect first
                    logger.info("Attempting to connect...")
                    if await self.market_hub.connect():
                        logger.info("Connected successfully!")
                        
                        # Critical: Keep the connection active immediately
                        # Subscribe right away to prevent disconnection
                        logger.info(f"Subscribing to {contract_id} immediately...")
                        await self.market_hub.subscribe_contract(contract_id)
                        
                        # Small delay to ensure subscription is processed
                        await asyncio.sleep(0.2)
                        
                        # Check if still connected
                        if self.market_hub.client.running:
                            self._running = True
                            self._connected_event.set()
                            logger.info("Market data manager ready and subscribed")
                            
                            while self._running:
                                if not self.market_hub.client.running:
                                    logger.warning("WebSocket lost – reconnecting…")
                                    # this will trigger the backoff loop above
                                    await self.market_hub.client.connect()
                                    # re-subscribe to your symbol
                                    await self.market_hub.subscribe_contract(contract_id)
                                await asyncio.sleep(0.05)
                        else:
                            logger.error("Connection lost after subscribe")
                            self._connected_event.set()
                    else:
                        logger.error("Failed to connect to market hub")
                        self._connected_event.set()
                        
                except Exception as e:
                    logger.error(f"Error in async_main: {e}", exc_info=True)
                    self._connected_event.set()
            
            try:
                self.loop.run_until_complete(async_main())
            except Exception as e:
                logger.error(f"Async loop error: {e}", exc_info=True)
                self._running = False
                self._connected_event.set()
        
        self.thread = threading.Thread(target=run_async_loop, daemon=True)
        self.thread.start()
        
        # Wait for connection
        if self._connected_event.wait(timeout=10):  # 10 second timeout
            if self._running:
                logger.info("Market data manager started successfully")
                return
            else:
                raise RuntimeError("Failed to connect to market hub")
        else:
            raise RuntimeError("Connection timeout")
    
    def add_connection_handler(self, callback: Callable[[bool], None]):
        """
        Register a callback to be invoked with a single bool argument:
        True when connected, False when disconnected.
        """
        if self.market_hub:
            # already built the async hub—attach immediately
            self.market_hub.add_connection_callback(callback)
        else:
            # stash for when start() builds the hub
            self._pending_callbacks['connection'].append(callback)

    @property
    def is_connected(self):
        """Check if the market hub is connected."""
        return self._running and self.market_hub and self.market_hub.client.running

    def stop(self):
        """Stop market data streaming."""
        self._running = False
        if self.loop and self.market_hub:
            future = asyncio.run_coroutine_threadsafe(
                self.market_hub.disconnect(), 
                self.loop
            )
            future.result(timeout=5)
    
    def add_bar_handler(self, callback: Callable):
        """Add a bar completion handler."""
        if self.market_hub:
            self.market_hub.add_bar_callback(callback)
        else:
            self._pending_callbacks['bar'].append(callback)
    
    def add_quote_handler(self, callback: Callable):
        """Add a quote handler."""
        if self.market_hub:
            self.market_hub.add_quote_callback(callback)
        else:
            self._pending_callbacks['quote'].append(callback)
    
    def add_trade_handler(self, callback: Callable):
        """Add a trade handler."""
        if self.market_hub:
            self.market_hub.add_trade_callback(callback)
        else:
            self._pending_callbacks['trade'].append(callback)
    
    def get_latest_bar(self, contract_id: str) -> Optional[Dict]:
        """Get latest bar for a contract."""
        if self.market_hub:
            return self.market_hub.get_latest_bar(contract_id)
        return None
    
    def get_latest_quote(self, contract_id: str) -> Optional[Dict]:
        """Get latest quote for a contract."""
        if self.market_hub:
            return self.market_hub.get_latest_quote(contract_id)
        return None


# Example usage
async def main():
    # Your token
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6IjY4NjA2IiwiaHR0cDovL3NjaGVtYXMueG1sc29hcC5vcmcvd3MvMjAwNS8wNS9pZGVudGl0eS9jbGFpbXMvc2lkIjoiZjkxZDBmMjctNDJmNC00MWMxLTgwNDctZWVlZWMwMDgwYTBiIiwiaHR0cDovL3NjaGVtYXMueG1sc29hcC5vcmcvd3MvMjAwNS8wNS9pZGVudGl0eS9jbGFpbXMvbmFtZSI6InBlbHQ4ODg1IiwiaHR0cDovL3NjaGVtYXMubWljcm9zb2Z0LmNvbS93cy8yMDA4LzA2L2lkZW50aXR5L2NsYWltcy9yb2xlIjoidXNlciIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvYXV0aGVudGljYXRpb25tZXRob2QiOiJhcGkta2V5IiwibXNkIjpbIkNNRUdST1VQX1RPQiIsIkNNRV9UT0IiXSwibWZhIjoidmVyaWZpZWQiLCJleHAiOjE3NTAyOTk0Njh9.5N8Ex8fHuC8O13qwONdNthhpwL22xboPDDq8OdGuxXg"
    SYMBOL = "CON.F.US.MNQ.U25"
    
    hub = TopStepMarketHub(TOKEN)
    
    # Define event handlers
    def on_bar(bar_data):
        print(f"New bar: {bar_data}")
    
    async def on_quote(contract_id, data):
        bid = data.get('bestBid') if data else None
        ask = data.get('bestAsk') if data else None
        last = data.get('lastPrice') if data else None
        print(f"Quote for {contract_id}: Bid={bid} Ask={ask} Last={last}")
    
    async def on_trade(contract_id, data):
        print(f"Trade for {contract_id}: Price={data.get('price')} Volume={data.get('volume')}")
    
    # Register handlers
    hub.add_bar_callback(on_bar)
    hub.add_quote_callback(on_quote)
    hub.add_trade_callback(on_trade)
    
    # Connect and subscribe
    if await hub.connect():
        print("✅ Connected successfully!")
        await hub.subscribe_contract(SYMBOL)
        
        # Run for 60 seconds
        await asyncio.sleep(60)
        
        # Cleanup
        await hub.disconnect()
    else:
        print("❌ Failed to connect")


if __name__ == "__main__":
    asyncio.run(main())