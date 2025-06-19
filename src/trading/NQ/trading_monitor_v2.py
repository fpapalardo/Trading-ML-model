"""
Enhanced Trading Monitor Dashboard

Real-time monitoring dashboard for the live trading system.
Displays system status, recent trades, performance metrics, and SignalR connection status.
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

from projectx_connector import ProjectXClient
from order_manager import OrderManager, OrderStatus

# Try to import from live_trading module
try:
    from live_trading import state, CONTRACT_SYMBOL
except ImportError:
    state = None
    CONTRACT_SYMBOL = None


class TradingMonitor:
    """Enhanced monitor for live trading system."""
    
    def __init__(self, px_client: ProjectXClient, account_id: int, order_manager: Optional[OrderManager] = None):
        self.px = px_client
        self.account_id = account_id
        self.order_manager = order_manager
        self.console = Console()
        
        # Timezone
        self.ny_tz = ZoneInfo("America/New_York")
        
        # Data storage
        self.orders_history = deque(maxlen=100)
        self.filled_orders = []
        self.system_status = {
            'signalr_status': 'Unknown',
            'last_bar': None,
            'last_signal': None,
            'last_quote': None,
            'uptime': datetime.now(self.ny_tz),
            'errors': 0,
            'bars_received': 0,
            'quotes_received': 0
        }
        
        # Performance metrics
        self.performance = {
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        # Account info cache
        self.account_info = None
        self.last_account_update = None
        
        # Update tracking
        self.last_orders_update = None
        self._running = True
        
    def update_system_status(self):
        """Update system status from global state."""
        if state and hasattr(state, 'market_hub'):
            # Update SignalR status
            if state.market_hub and state.market_hub.is_connected:
                self.system_status['signalr_status'] = 'Connected'
            else:
                self.system_status['signalr_status'] = 'Disconnected'
            
            # Update last bar timestamp
            if state.last_bar_timestamp:
                self.system_status['last_bar'] = state.last_bar_timestamp
            
            # Get latest quote
            if CONTRACT_SYMBOL and state.market_hub:
                quote = state.market_hub.get_latest_quote(CONTRACT_SYMBOL)
                if quote:
                    self.system_status['last_quote'] = {
                        'bid': quote.get('bestBid'),
                        'ask': quote.get('bestAsk'),
                        'last': quote.get('lastPrice'),
                        'time': datetime.now(self.ny_tz)
                    }
    
    def update_orders(self):
        """Update orders and calculate performance metrics."""
        try:
            # Get recent orders (last 24 hours)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=24)
            
            orders = self.px.search_orders(start=start_time, end=end_time)
            
            # Process orders
            for order in orders:
                order_id = order.get('id') or order.get('orderId')
                
                # Skip if already processed
                if any(o.get('id') == order_id for o in self.orders_history):
                    continue
                
                # Extract order info
                order_info = {
                    'id': order_id,
                    'time': self._parse_timestamp(order.get('placedTimestamp')),
                    'side': 'BUY' if order.get('side') == 0 else 'SELL',
                    'type': self._get_order_type(order.get('type')),
                    'status': self._get_order_status(order.get('status')),
                    'quantity': order.get('size', 0),
                    'limit_price': order.get('limitPrice'),
                    'stop_price': order.get('stopPrice'),
                    'fill_price': order.get('averageFillPrice'),
                    'filled_qty': order.get('filledSize', 0),
                    'commission': order.get('commission', 0),
                    'pnl': 0.0  # Calculate later
                }
                
                self.orders_history.append(order_info)
                
                # Track filled orders for P&L calculation
                if order_info['status'] == 'Filled':
                    self.filled_orders.append(order_info)
            
            # Calculate performance metrics
            self._calculate_performance()
            
            self.last_orders_update = datetime.now()
            
        except Exception as e:
            self.system_status['errors'] += 1
            self.console.print(f"[red]Error updating orders: {e}[/red]")
    
    def update_account(self):
        """Update account information."""
        try:
            # Only update every 30 seconds
            if (self.last_account_update and 
                (datetime.now() - self.last_account_update).seconds < 30):
                return
            
            # Search for active accounts
            accounts = self.px._session.post(
                self.px.urls['accounts'],
                json={"onlyActiveAccounts": True},
                headers=self.px._headers(),
                timeout=self.px.request_timeout
            ).json().get("accounts", [])
            
            # Find our account
            self.account_info = next(
                (acc for acc in accounts if acc['id'] == self.account_id),
                None
            )
            
            self.last_account_update = datetime.now()
            
        except Exception as e:
            self.system_status['errors'] += 1
            self.console.print(f"[red]Error updating account: {e}[/red]")
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp string to datetime."""
        if not ts_str:
            return None
        try:
            # Parse ISO format and convert to NY timezone
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            return dt.astimezone(self.ny_tz)
        except:
            return None
    
    def _get_order_type(self, type_code: int) -> str:
        """Convert order type code to string."""
        types = {
            1: "Market",
            2: "Limit", 
            3: "Stop",
            4: "StopLimit"
        }
        return types.get(type_code, "Unknown")
    
    def _get_order_status(self, status_code: int) -> str:
        """Convert order status code to string."""
        statuses = {
            0: "Working",
            1: "Filled",
            2: "Cancelled",
            3: "Expired",
            4: "Rejected"
        }
        return statuses.get(status_code, "Unknown")
    
    def _calculate_performance(self):
        """Calculate performance metrics from filled orders."""
        if not self.filled_orders:
            return
        
        # Group orders by entry/exit pairs
        # This is simplified - in reality you'd need to match entries with exits
        today = datetime.now(self.ny_tz).date()
        today_trades = []
        
        # Calculate daily P&L (simplified)
        daily_pnl = 0.0
        total_pnl = 0.0
        
        for order in self.filled_orders:
            if order['time'] and order['time'].date() == today:
                # This is a simplified calculation
                # In reality, you'd match buy/sell pairs
                if order.get('fill_price'):
                    # Add to daily trades
                    today_trades.append(order)
        
        # Update performance metrics
        self.performance['total_trades'] = len(self.filled_orders) // 2  # Rough estimate
        self.performance['daily_pnl'] = daily_pnl
        self.performance['total_pnl'] = total_pnl
        
        # Calculate win rate if we have trades
        if self.performance['total_trades'] > 0:
            self.performance['win_rate'] = (
                self.performance['winning_trades'] / self.performance['total_trades'] * 100
            )
    
    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        # Header with timestamp
        header_text = Text()
        header_text.append("🤖 LIVE TRADING MONITOR", style="bold cyan")
        header_text.append(f"  |  ", style="dim")
        header_text.append(f"{datetime.now(self.ny_tz):%Y-%m-%d %H:%M:%S ET}", style="yellow")
        
        header = Panel(
            header_text,
            box=box.DOUBLE,
            padding=(0, 1)
        )
        
        # System Status Panel
        uptime = datetime.now(self.ny_tz) - self.system_status['uptime']
        
        # Format last quote info
        quote_info = "N/A"
        if self.system_status['last_quote']:
            q = self.system_status['last_quote']
            quote_info = f"Bid: {q['bid']:.2f}  Ask: {q['ask']:.2f}  Last: {q['last']:.2f}"
        
        status_text = f"""
SignalR Status: {self._format_status(self.system_status['signalr_status'])}
Uptime: {self._format_timedelta(uptime)}
Last Bar: {self.system_status['last_bar'].strftime('%H:%M:%S') if self.system_status['last_bar'] else 'N/A'}
Market Quote: {quote_info}
Errors: {self.system_status['errors']}
"""
        
        status_panel = Panel(
            status_text.strip(),
            title="System Status",
            title_align="left",
            border_style="green" if self.system_status['signalr_status'] == 'Connected' else "red"
        )
        
        # Account Info Panel
        if self.account_info:
            balance = self.account_info.get('balance', 0)
            buying_power = self.account_info.get('buyingPower', balance)
            
            account_text = f"""
Account: {self.account_info.get('name', 'Unknown')}
Balance: ${balance:,.2f}
Buying Power: ${buying_power:,.2f}
Daily P&L: ${self.performance['daily_pnl']:+,.2f}
Total P&L: ${self.performance['total_pnl']:+,.2f}
"""
        else:
            account_text = "Loading account info..."
        
        account_panel = Panel(
            account_text.strip(),
            title="Account Info",
            title_align="left",
            border_style="blue"
        )
        
        # Performance Metrics Panel
        perf_text = f"""
Win Rate: {self.performance['win_rate']:.1f}%
Total Trades: {self.performance['total_trades']}
Winning: {self.performance['winning_trades']} | Losing: {self.performance['losing_trades']}
Avg Win: ${self.performance['avg_win']:.2f} | Avg Loss: ${self.performance['avg_loss']:.2f}
Best: ${self.performance['largest_win']:.2f} | Worst: ${self.performance['largest_loss']:.2f}
"""
        
        perf_panel = Panel(
            perf_text.strip(),
            title="Performance Metrics",
            title_align="left",
            border_style="magenta"
        )
        
        # Active Orders Table
        if self.order_manager:
            active_groups = self.order_manager.get_active_groups()
            active_orders_text = f"Active Order Groups: {len(active_groups)}\n"
            for group in active_groups[:3]:  # Show first 3
                active_orders_text += f"  Entry: {group.entry_order_id} | TP: {group.tp_order_id} | SL: {group.sl_order_id}\n"
        else:
            active_orders_text = "Order manager not connected"
        
        active_panel = Panel(
            active_orders_text.strip(),
            title="Active Orders",
            title_align="left",
            border_style="yellow"
        )
        
        # Recent Orders Table
        orders_table = Table(title="Recent Orders", box=box.SIMPLE)
        orders_table.add_column("Time", style="cyan", width=8)
        orders_table.add_column("Type", style="white", width=8)
        orders_table.add_column("Side", style="magenta", width=4)
        orders_table.add_column("Qty", style="white", width=3)
        orders_table.add_column("Price", style="yellow", width=8)
        orders_table.add_column("Status", style="green", width=10)
        
        # Add last 15 orders
        for order in list(self.orders_history)[-15:]:
            if not order.get('time'):
                continue
                
            # Color code by status
            status_style = {
                'Filled': 'green',
                'Working': 'yellow',
                'Cancelled': 'red',
                'Rejected': 'red bold'
            }.get(order['status'], 'white')
            
            # Determine price to show
            price = order.get('fill_price') or order.get('limit_price') or order.get('stop_price') or 0
            
            orders_table.add_row(
                order['time'].strftime("%H:%M:%S"),
                order['type'],
                order['side'],
                str(order['quantity']),
                f"${price:.2f}" if price else "Market",
                Text(order['status'], style=status_style)
            )
        
        orders_panel = Panel(orders_table, title="Order History", title_align="left")
        
        # Arrange layout
        layout.split_column(
            Layout(header, size=3),
            Layout(name="top", size=12),
            Layout(name="middle", size=8),
            Layout(name="bottom")
        )
        
        layout["top"].split_row(
            status_panel,
            account_panel,
            perf_panel
        )
        
        layout["middle"].update(active_panel)
        layout["bottom"].update(orders_panel)
        
        return layout
    
    def _format_status(self, status: str) -> str:
        """Format status with color."""
        if status == "Connected":
            return f"[green]● {status}[/green]"
        elif status == "Disconnected":
            return f"[red]● {status}[/red]"
        else:
            return f"[yellow]● {status}[/yellow]"
    
    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def run(self):
        """Run the monitoring dashboard."""
        # Initial updates
        self.update_account()
        self.update_orders()
        
        with Live(self.create_layout(), refresh_per_second=1) as live:
            update_counter = 0
            
            while self._running:
                try:
                    # Update system status every second
                    self.update_system_status()
                    
                    # Update orders every 5 seconds
                    if update_counter % 5 == 0:
                        self.update_orders()
                    
                    # Update account every 30 seconds
                    if update_counter % 30 == 0:
                        self.update_account()
                    
                    # Update display
                    live.update(self.create_layout())
                    
                    # Increment counter and sleep
                    update_counter += 1
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Dashboard error: {e}[/red]")
                    time.sleep(1)
        
        self.console.print("\n[yellow]Monitor stopped[/yellow]")


def main():
    """Main function to run the monitor standalone."""
    console = Console()
    
    try:
        # Check if we can import from live trading
        if state is None:
            console.print("[red]Cannot import from live_trading module[/red]")
            console.print("[yellow]Make sure live_trading.py is running first[/yellow]")
            return
        
        # Wait for live trading to initialize
        console.print("[cyan]Waiting for live trading system to initialize...[/cyan]")
        timeout = 30
        start_time = time.time()

        while not hasattr(state, 'is_initialized') or not state.is_initialized:
            if time.time() - start_time > timeout:
                console.print("[red]Timeout waiting for initialization[/red]")
                console.print("[yellow]Make sure live_trading.py is running and has completed initialization[/yellow]")
                return
            time.sleep(1)

        # Give it a moment more to ensure everything is ready
        time.sleep(2)

        # Verify we have what we need
        if not hasattr(state, 'px') or state.px is None:
            console.print("[red]State initialized but ProjectX client not found[/red]")
            return
        
        console.print("[green]Live trading system detected![/green]")
        
        # Create monitor using the live trading state
        monitor = TradingMonitor(
            state.px,
            state.px.account_id,
            order_manager=getattr(state, 'order_manager', None)
        )
        
        # Run the monitor
        monitor.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()