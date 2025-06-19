"""
Telegram Trading Notification System

Sends real-time trading alerts and system status updates to Telegram.
Requires python-telegram-bot library: pip install python-telegram-bot

To set up:
1. Create a bot with @BotFather on Telegram
2. Get your bot token
3. Get your chat ID by messaging the bot and visiting:
   https://api.telegram.org/bot<YourBOTToken>/getUpdates
"""

import asyncio
import threading
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict
from queue import Queue, Empty
import json
import time

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Warning: python-telegram-bot not installed. Run: pip install python-telegram-bot")

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends trading notifications to Telegram.
    
    Features:
    - Trade signals and executions
    - System status updates
    - Error alerts
    - Daily performance summaries
    - Market quotes on demand
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot library not installed")
            
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        
        # Message queue for async sending
        self.message_queue = Queue()
        self._running = False
        self._send_thread = None
        
        # Rate limiting
        self.min_message_interval = 1.0  # seconds between messages
        self.last_message_time = 0
        
        # Message formatting
        self.ny_tz = ZoneInfo("America/New_York")
        
        # Emoji mappings
        self.emoji = {
            'buy': '🟢',
            'sell': '🔴',
            'tp': '✅',
            'sl': '❌',
            'warning': '⚠️',
            'error': '🚨',
            'info': 'ℹ️',
            'money': '💰',
            'chart': '📊',
            'robot': '🤖',
            'time': '⏰',
            'connected': '🟢',
            'disconnected': '🔴',
            'rocket': '🚀'
        }
    
    def _run_loop(self):
        """Entry point for the asyncio event loop thread."""
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _ensure_loop(self):
        """Create & start the loop thread once."""
        if not hasattr(self, "_async_loop"):
            self._async_loop = asyncio.new_event_loop()
            t = threading.Thread(target=self._run_loop, daemon=True)
            t.start()
            
    def start(self):
        """Start the message sending thread."""
        if self._running:
            return
            
        self._ensure_loop()
        
        self._running = True
        self._send_thread = threading.Thread(
            target=self._send_loop,
            daemon=True
        )
        self._send_thread.start()
        
        # Send startup message
        self.send_message(
            f"{self.emoji['robot']} Trading Bot Started\n"
            f"{self.emoji['time']} {datetime.now(self.ny_tz):%Y-%m-%d %H:%M:%S ET}"
        )
        
        logger.info("Telegram notifier started")
    
    def stop(self):
        """Stop the message sending thread."""
        if not self._running:
            return
            
        # Send shutdown message
        self.send_message(
            f"{self.emoji['robot']} Trading Bot Stopped\n"
            f"{self.emoji['time']} {datetime.now(self.ny_tz):%Y-%m-%d %H:%M:%S ET}"
        )
        
        self._running = False
        if self._send_thread:
            self._send_thread.join(timeout=5)
            
        logger.info("Telegram notifier stopped")
    
    def _send_loop(self):
        """Message sending loop that runs in separate thread."""
        while self._running:
            try:
                # Get message from queue (timeout allows checking _running)
                message = self.message_queue.get(timeout=1)
                
                # Rate limiting
                time_since_last = time.time() - self.last_message_time
                if time_since_last < self.min_message_interval:
                    time.sleep(self.min_message_interval - time_since_last)
                
                # Send message
                self._send_telegram_message(message)
                self.last_message_time = time.time()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in Telegram send loop: {e}")
    
    def _send_telegram_message(self, message: str):
        """Send a message to Telegram (synchronous)."""
        try:
            coro = self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
            # wait for it (or you could omit .result() if you don't care)
            future.result()
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
        
    
    def send_message(self, message: str, priority: bool = False):
        """
        Queue a message for sending.
        
        Args:
            message: Message text (can include HTML formatting)
            priority: If True, add to front of queue
        """
        if priority:
            # For high priority, send immediately in new thread
            threading.Thread(
                target=self._send_telegram_message,
                args=(message,),
                daemon=True
            ).start()
        else:
            self.message_queue.put(message)
    
    def send_trade_signal(self, side: str, price: float, tp: float, sl: float, bar_time: datetime):
        """Send trade signal notification."""
        emoji = self.emoji['buy'] if side.upper() == 'BUY' else self.emoji['sell']
        
        # Calculate point differences
        tp_points = abs(tp - price)
        sl_points = abs(sl - price)
        risk_reward = tp_points / sl_points if sl_points > 0 else 0
        
        message = (
            f"<b>{emoji} {side.upper()} SIGNAL</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Entry: <b>${price:.2f}</b>\n"
            f"TP: ${tp:.2f} (+{tp_points:.2f} pts)\n"
            f"SL: ${sl:.2f} (-{sl_points:.2f} pts)\n"
            f"R:R Ratio: <b>{risk_reward:.1f}</b>\n"
            f"Bar Time: {bar_time:%H:%M:%S ET}\n"
            f"Signal Time: {datetime.now(self.ny_tz):%H:%M:%S ET}"
        )
        
        self.send_message(message, priority=True)
    
    def send_order_filled(self, order_type: str, fill_price: float, quantity: int):
        """Send order filled notification."""
        if order_type.upper() == 'TP':
            emoji = self.emoji['tp']
            title = "TAKE PROFIT HIT"
        elif order_type.upper() == 'SL':
            emoji = self.emoji['sl']
            title = "STOP LOSS HIT"
        else:
            emoji = self.emoji['money']
            title = "ORDER FILLED"
        
        message = (
            f"<b>{emoji} {title}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Fill Price: <b>${fill_price:.2f}</b>\n"
            f"Quantity: {quantity}\n"
            f"Time: {datetime.now(self.ny_tz):%H:%M:%S ET}"
        )
        
        self.send_message(message, priority=True)
    
    def send_connection_status(self, connected: bool, service: str = "SignalR"):
        """Send connection status update."""
        if connected:
            emoji = self.emoji['connected']
            status = "CONNECTED"
        else:
            emoji = self.emoji['disconnected']
            status = "DISCONNECTED"
        
        message = (
            f"{emoji} <b>{service} {status}</b>\n"
            f"Time: {datetime.now(self.ny_tz):%H:%M:%S ET}"
        )
        
        self.send_message(message)
    
    def send_error(self, error_message: str, error_type: str = "System"):
        """Send error notification."""
        message = (
            f"{self.emoji['error']} <b>{error_type} ERROR</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{error_message}\n"
            f"Time: {datetime.now(self.ny_tz):%H:%M:%S ET}"
        )
        
        self.send_message(message, priority=True)
    
    def send_daily_summary(self, stats: Dict):
        """Send daily performance summary."""
        pnl = stats.get('daily_pnl', 0)
        pnl_emoji = self.emoji['money'] if pnl >= 0 else self.emoji['warning']
        
        message = (
            f"{self.emoji['chart']} <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Date: {datetime.now(self.ny_tz):%Y-%m-%d}\n"
            f"P&L: {pnl_emoji} <b>${pnl:+,.2f}</b>\n"
            f"Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1f}%\n"
            f"Wins: {stats.get('winning_trades', 0)} | Losses: {stats.get('losing_trades', 0)}\n"
            f"Avg Win: ${stats.get('avg_win', 0):.2f}\n"
            f"Avg Loss: ${stats.get('avg_loss', 0):.2f}\n"
            f"Best Trade: ${stats.get('best_trade', 0):.2f}\n"
            f"Worst Trade: ${stats.get('worst_trade', 0):.2f}"
        )
        
        self.send_message(message)
    
    def send_market_update(self, bid: float, ask: float, last: float):
        """Send current market quote."""
        spread = ask - bid if bid and ask else 0
        
        message = (
            f"{self.emoji['chart']} <b>MARKET UPDATE</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Bid: ${bid:.2f}\n"
            f"Ask: ${ask:.2f}\n"
            f"Last: ${last:.2f}\n"
            f"Spread: ${spread:.2f}\n"
            f"Time: {datetime.now(self.ny_tz):%H:%M:%S ET}"
        )
        
        self.send_message(message)


def create_telegram_notifier(config_file: str = "telegram_config.json") -> Optional[TelegramNotifier]:
    """
    Create Telegram notifier from config file.
    
    Args:
        config_file: Path to JSON config file with bot_token and chat_id
        
    Returns:
        TelegramNotifier instance or None if config not found
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        bot_token = config.get('bot_token')
        chat_id = config.get('chat_id')
        
        if not bot_token or not chat_id:
            logger.error("Missing bot_token or chat_id in config")
            return None
            
        notifier = TelegramNotifier(bot_token, chat_id)
        return notifier
        
    except FileNotFoundError:
        logger.warning(f"Telegram config file not found: {config_file}")
        logger.info("Create a file with: {\"bot_token\": \"YOUR_BOT_TOKEN\", \"chat_id\": \"YOUR_CHAT_ID\"}")
        return None
    except Exception as e:
        logger.error(f"Failed to create Telegram notifier: {e}")
        return None


# Integration functions for live_trading.py

def setup_telegram_notifications(notifier: TelegramNotifier, state, order_manager: Optional['OrderManager'] = None):
    """
    Set up Telegram notifications for the live trading system.
    
    Args:
        notifier: TelegramNotifier instance
        state: Trading state object
        order_manager: Order manager instance
    """
    # Start the notifier
    notifier.start()
    
    # Hook into order manager callbacks if available
    if order_manager:
        # Order filled callback
        def on_order_filled(order_id: int, order_type: str, group):
            # Get fill details from recent orders
            try:
                orders = state.px.search_orders()
                order = next((o for o in orders if o.get('id') == order_id), None)
                if order:
                    fill_price = order.get('averageFillPrice', 0)
                    quantity = order.get('filledSize', 0)
                    notifier.send_order_filled(order_type, fill_price, quantity)
            except Exception as e:
                logger.error(f"Error sending order fill notification: {e}")
        
        order_manager.on_order_filled = on_order_filled
        
        # Error callback
        def on_order_error(error):
            notifier.send_error(str(error), "Order")
        
        order_manager.on_order_error = on_order_error
    
    # Hook into market hub connection status if available
    if hasattr(state, 'market_hub') and state.market_hub:
        async def on_connection_change(connected: bool):
            notifier.send_connection_status(connected, "Market Data")
        
        state.market_hub.add_connection_handler(on_connection_change)
    
    # Store notifier in state for access from signal processing
    state.telegram_notifier = notifier
    
    logger.info("Telegram notifications configured")


# Example usage in live_trading.py:
"""
# In your main() function after initializing state:

# Set up Telegram notifications (optional)
telegram_notifier = create_telegram_notifier()
if telegram_notifier:
    setup_telegram_notifications(telegram_notifier, state, state.order_manager)
    print("✓ Telegram notifications: Enabled")
else:
    print("✗ Telegram notifications: Disabled")

# In process_new_bar_signal() when placing order:

if hasattr(state, 'telegram_notifier') and state.telegram_notifier:
    state.telegram_notifier.send_trade_signal(side, price, tp_price, sl_price, bar_time)
"""