"""
Live Trading Monitor Dashboard

Real-time monitoring dashboard for the tsxapi4py-based trading system.
Displays system status, recent trades, and performance metrics.
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box

from tsxapipy import APIClient, authenticate, api_schemas


class TradingMonitor:
    """Monitor for live trading system."""
    
    def __init__(self, api_client: APIClient, account_id: int):
        self.api_client = api_client
        self.account_id = account_id
        self.console = Console()
        
        # Data storage
        self.orders_history = deque(maxlen=50)
        self.trades_today = []
        self.system_status = {
            'stream_status': 'Unknown',
            'last_bar': None,
            'last_signal': None,
            'uptime': datetime.now(),
            'errors': 0
        }
        
        # Performance metrics
        self.daily_pnl = 0.0
        self.win_rate = 0.0
        self.total_trades = 0
        
    def update_data(self):
        """Update all monitoring data."""
        try:
            # Get today's orders
            end_time = datetime.now()
            start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            orders = self.api_client.get_orders(
                account_id=self.account_id,
                start_timestamp=start_time,
                end_timestamp=end_time
            )
            
            # Update orders history
            for order in orders:
                order_dict = {
                    'id': order.id,
                    'time': order.placed_timestamp,
                    'side': 'BUY' if order.side == 0 else 'SELL',
                    'status': self._get_order_status(order),
                    'price': order.limit_price or order.stop_price,
                    'pnl': self._calculate_order_pnl(order)
                }
                
                # Add to history if not already there
                if not any(o['id'] == order_dict['id'] for o in self.orders_history):
                    self.orders_history.append(order_dict)
            
            # Calculate daily metrics
            self._calculate_daily_metrics()
            
            # Get account info
            accounts = self.api_client.get_accounts(only_active=True)
            self.account_info = next(
                (acc for acc in accounts if acc.id == self.account_id),
                None
            )
            
        except Exception as e:
            self.system_status['errors'] += 1
            self.console.print(f"[red]Error updating data: {e}[/red]")
    
    def _get_order_status(self, order: api_schemas.OrderDetails) -> str:
        """Get human-readable order status."""
        status_map = {
            0: "Working",
            1: "Filled",
            2: "Cancelled",
            3: "Expired",
            4: "Rejected"
        }
        return status_map.get(order.status, "Unknown")
    
    def _calculate_order_pnl(self, order: api_schemas.OrderDetails) -> float:
        """Calculate P&L for an order."""
        # This is simplified - you'd need to match with fills
        if order.status == 1:  # Filled
            # Would need to get fill price and exit price
            return 0.0  # Placeholder
        return 0.0
    
    def _calculate_daily_metrics(self):
        """Calculate daily performance metrics."""
        if not self.orders_history:
            return
            
        # Filter today's completed trades
        today_trades = [
            o for o in self.orders_history 
            if o['status'] in ['Filled', 'Cancelled'] and 
            o['time'].date() == datetime.now().date()
        ]
        
        self.total_trades = len(today_trades)
        
        if self.total_trades > 0:
            # Calculate win rate
            winning_trades = sum(1 for t in today_trades if t['pnl'] > 0)
            self.win_rate = (winning_trades / self.total_trades) * 100
            
            # Calculate total P&L
            self.daily_pnl = sum(t['pnl'] for t in today_trades)
    
    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        # Header
        header = Panel(
            Text("🤖 LIVE TRADING MONITOR", style="bold cyan", justify="center"),
            box=box.DOUBLE
        )
        
        # System Status Panel
        uptime = datetime.now() - self.system_status['uptime']
        status_text = f"""
Stream Status: {self._format_status(self.system_status['stream_status'])}
Uptime: {self._format_timedelta(uptime)}
Last Bar: {self.system_status['last_bar'] or 'N/A'}
Last Signal: {self.system_status['last_signal'] or 'N/A'}
Errors: {self.system_status['errors']}
"""
        status_panel = Panel(
            status_text.strip(),
            title="System Status",
            title_align="left",
            border_style="green" if self.system_status['errors'] == 0 else "yellow"
        )
        
        # Account Info Panel
        if hasattr(self, 'account_info') and self.account_info:
            account_text = f"""
Account: {self.account_info.name}
Balance: ${self.account_info.balance:,.2f}
Daily P&L: ${self.daily_pnl:+,.2f}
Win Rate: {self.win_rate:.1f}%
Trades Today: {self.total_trades}
"""
        else:
            account_text = "Loading account info..."
            
        account_panel = Panel(
            account_text.strip(),
            title="Account Performance",
            title_align="left",
            border_style="blue"
        )
        
        # Recent Orders Table
        orders_table = Table(title="Recent Orders", box=box.SIMPLE)
        orders_table.add_column("Time", style="cyan")
        orders_table.add_column("Side", style="magenta")
        orders_table.add_column("Price", style="yellow")
        orders_table.add_column("Status", style="green")
        orders_table.add_column("P&L", style="white")
        
        # Add last 10 orders
        for order in list(self.orders_history)[-10:]:
            status_style = "green" if order['status'] == "Filled" else "yellow"
            pnl_style = "green" if order['pnl'] > 0 else "red" if order['pnl'] < 0 else "white"
            
            orders_table.add_row(
                order['time'].strftime("%H:%M:%S"),
                order['side'],
                f"${order['price']:.2f}" if order['price'] else "N/A",
                Text(order['status'], style=status_style),
                Text(f"${order['pnl']:+.2f}", style=pnl_style)
            )
        
        orders_panel = Panel(orders_table, title="Order History", title_align="left")
        
        # Arrange layout
        layout.split_column(
            header,
            Layout(name="top"),
            Layout(name="bottom")
        )
        
        layout["top"].split_row(
            status_panel,
            account_panel
        )
        
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
        with Live(self.create_layout(), refresh_per_second=1) as live:
            while True:
                try:
                    # Update data
                    self.update_data()
                    
                    # Update display
                    live.update(self.create_layout())
                    
                    # Wait before next update
                    time.sleep(5)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Dashboard error: {e}[/red]")
                    time.sleep(5)


def main():
    """Main function to run the monitor."""
    console = Console()
    
    try:
        # Authenticate
        console.print("[cyan]Authenticating with TopStepX...[/cyan]")
        token, acquired_at = authenticate()
        
        # Create API client
        api_client = APIClient(initial_token=token, token_acquired_at=acquired_at)
        
        # Get account
        accounts = api_client.get_accounts(only_active=True)
        if not accounts:
            console.print("[red]No active accounts found![/red]")
            return
            
        # Let user select account or use first one
        account = accounts[0]
        console.print(f"[green]Using account: {account.name} (ID: {account.id})[/green]")
        
        # Create and run monitor
        monitor = TradingMonitor(api_client, account.id)
        monitor.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    # Install rich if not available
    try:
        import rich
    except ImportError:
        print("Installing rich for dashboard display...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    
    main()