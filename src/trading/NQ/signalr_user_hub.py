# signalr_user_hub.py - Fixed version with minimal changes
import asyncio
import logging
from typing import Callable, Set

from signalr_market_hub import ReconnectingSignalRClient

logger = logging.getLogger(__name__)

class UserHubClient:
    """SignalR client for user‐specific updates: accounts, orders, positions, trades."""

    def __init__(self, token: str, account_id: int):
        self.token = token
        self.account_id = account_id
        self.base_url = "wss://rtc.topstepx.com/hubs/user"
        self.client = self._build_signalr_client()
        self.subscribed = False
        self._connected = False

    def _build_signalr_client(self) -> ReconnectingSignalRClient:
        # Reuse the same class you have for market, just point at /user
        cli = ReconnectingSignalRClient(self.token)
        cli.base_url = self.base_url
        # Handlers
        cli.callbacks.update({
            'GatewayUserAccount': [],
            'GatewayUserOrder':   [],
            'GatewayUserPosition':[],
            'GatewayUserTrade':   []
        })
        # Wire up reconnection resubscribe
        cli._reconnect_interval = 2
        cli.max_reconnect_interval = 30

        return cli

    async def start(self):
        """Open connection and subscribe to user channels."""
        ok = await self.client.connect()
        if not ok:
            raise RuntimeError("Failed to connect to UserHub")
        
        self._connected = True
        await self._subscribe_all()
        logger.info("UserHubClient started")
        
        # Keep running until stopped
        while self.client.running and self._connected:
            await asyncio.sleep(1)
            
            # Check if we need to resubscribe after reconnection
            if self.client.running and not self.subscribed:
                await self._subscribe_all()

    async def stop(self):
        """Tear down the WS connection."""
        self._connected = False
        await self.client.disconnect()
        logger.info("UserHubClient stopped")

    async def _subscribe_all(self):
        """Invoke all subscribe methods; idempotent on reconnect."""
        try:
            # Account updates
            await self.client.invoke("SubscribeAccounts")
            # Orders / positions / trades for our account
            await self.client.invoke("SubscribeOrders",    self.account_id)
            await self.client.invoke("SubscribePositions", self.account_id)
            await self.client.invoke("SubscribeTrades",    self.account_id)
            self.subscribed = True
            logger.debug("UserHubClient subscriptions sent")
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            self.subscribed = False

    def on(self, event: str, callback: Callable):
        """
        Register a callback for one of:
        'GatewayUserAccount', 'GatewayUserOrder', 'GatewayUserPosition', 'GatewayUserTrade'
        
        The callback should be a regular function (not async) that accepts parameters.
        """
        if event in self.client.callbacks:
            # Wrap callbacks to handle sync/async properly
            async def async_wrapper(*args):
                try:
                    # Handle both sync and async callbacks
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args)
                    else:
                        callback(*args)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
                    
            self.client.callbacks[event].append(async_wrapper)
        else:
            raise ValueError(f"Unknown event: {event}")

    # Convenience wrappers for each event
    def on_account(self,    fn): self.on('GatewayUserAccount', fn)
    def on_order(self,      fn): self.on('GatewayUserOrder',   fn)
    def on_position(self,   fn): self.on('GatewayUserPosition',fn)
    def on_trade(self,      fn): self.on('GatewayUserTrade',   fn)
    
    async def _run_callback(self, callback, args):
        """Run callback, handling both sync and async functions."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")