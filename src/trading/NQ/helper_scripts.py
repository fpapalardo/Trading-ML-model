"""
Helper Scripts and Utilities for the Trading System

This file contains useful scripts for testing, monitoring, and managing
the live trading system.
"""

import os
import json
import time
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

# === 1. System Health Checker ===

def check_system_health():
    """
    Check if all components of the trading system are working.
    Run this before starting live trading.
    """
    print("=== Trading System Health Check ===\n")
    
    issues = []
    
    # Check imports
    print("Checking dependencies...")
    try:
        import websockets
        print("✓ websockets")
    except ImportError:
        print("✗ websockets - Run: pip install websockets")
        issues.append("websockets not installed")
    
    try:
        import telegram
        print("✓ telegram")
    except ImportError:
        print("✗ telegram - Run: pip install python-telegram-bot")
        issues.append("telegram not installed")
    
    try:
        import rich
        print("✓ rich")
    except ImportError:
        print("✗ rich - Run: pip install rich")
        issues.append("rich not installed")
    
    try:
        import joblib
        print("✓ joblib")
    except ImportError:
        print("✗ joblib - Run: pip install joblib")
        issues.append("joblib not installed")
    
    # Check config files
    print("\nChecking configuration files...")
    try:
        from config import FUTURES
        if FUTURES.get("topstep", {}).get("username"):
            print("✓ config.py found with credentials")
        else:
            print("✗ config.py missing credentials")
            issues.append("config.py missing credentials")
    except ImportError:
        print("✗ config.py not found")
        issues.append("config.py not found")
    
    # Check model file
    print("\nChecking model file...")
    import os
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if model_files:
        print(f"✓ Found model files: {', '.join(model_files)}")
    else:
        print("✗ No .pkl model files found")
        issues.append("No model file found")
    
    # Check Telegram config
    print("\nChecking Telegram configuration...")
    if os.path.exists('telegram_config.json'):
        try:
            with open('telegram_config.json') as f:
                tg_config = json.load(f)
            if tg_config.get('bot_token') and tg_config.get('chat_id'):
                print("✓ telegram_config.json found and valid")
            else:
                print("✗ telegram_config.json missing bot_token or chat_id")
                issues.append("telegram_config.json incomplete")
        except Exception as e:
            print(f"✗ telegram_config.json invalid: {e}")
            issues.append("telegram_config.json invalid")
    else:
        print("⚠ telegram_config.json not found (optional)")
    
    # Summary
    print("\n" + "="*40)
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All systems ready!")
        return True


# === 2. Order Cleanup Utility ===

def cancel_all_open_orders():
    """
    Emergency function to cancel all open orders.
    Use this if the system crashes with orders still open.
    """
    from projectx_connector import ProjectXClient
    from config import FUTURES
    
    print("=== Emergency Order Cancellation ===\n")
    
    # Initialize client
    px = ProjectXClient(
        FUTURES["topstep"]["username"],
        FUTURES["topstep"]["api_key"]
    )
    
    try:
        px.authenticate()
        print("✓ Authenticated successfully")
        
        # Get open orders
        open_orders = px.search_open_orders()
        
        if not open_orders:
            print("No open orders found.")
            return
        
        print(f"\nFound {len(open_orders)} open orders:")
        for order in open_orders:
            order_id = order.get('id') or order.get('orderId')
            side = 'BUY' if order.get('side') == 0 else 'SELL'
            price = order.get('limitPrice') or order.get('stopPrice') or 'Market'
            print(f"  - Order {order_id}: {side} @ {price}")
        
        # Confirm cancellation
        response = input("\nCancel all orders? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancellation aborted.")
            return
        
        # Cancel each order
        for order in open_orders:
            order_id = order.get('id') or order.get('orderId')
            try:
                px.cancel_order(order_id)
                print(f"✓ Cancelled order {order_id}")
            except Exception as e:
                print(f"✗ Failed to cancel order {order_id}: {e}")
                
    except Exception as e:
        print(f"Error: {e}")


# === 3. Market Data Tester ===

async def test_market_data(duration_seconds=30):
    """
    Test SignalR market data connection and display quotes.
    """
    from signalr_market_hub import TopStepMarketHub
    from projectx_connector import ProjectXClient
    from config import FUTURES
    import asyncio
    
    print(f"=== Market Data Test ({duration_seconds}s) ===\n")
    
    # Get token
    px = ProjectXClient(
        FUTURES["topstep"]["username"],
        FUTURES["topstep"]["api_key"]
    )
    px.authenticate()
    
    # Find contract
    contract_info = px.get_contract_info("MNQ")
    contract_symbol = contract_info['id']
    print(f"Testing contract: {contract_symbol}")
    
    # Create market hub
    hub = TopStepMarketHub(px.token)
    
    # Quote counter
    quote_count = 0
    
    async def on_quote(contract_id, data):
        nonlocal quote_count
        quote_count += 1
        if quote_count % 10 == 0:  # Print every 10th quote
            bid = data.get('bestBid', 'N/A')
            ask = data.get('bestAsk', 'N/A')
            print(f"Quote #{quote_count}: Bid={bid} Ask={ask}")
    
    hub.add_quote_callback(on_quote)
    
    # Connect and subscribe
    if await hub.connect():
        print("✓ Connected to market data")
        await hub.subscribe_contract(contract_symbol)
        print(f"✓ Subscribed to {contract_symbol}")
        
        # Run for specified duration
        print(f"\nReceiving quotes for {duration_seconds} seconds...")
        await asyncio.sleep(duration_seconds)
        
        print(f"\nTest complete. Received {quote_count} quotes.")
        await hub.disconnect()
    else:
        print("✗ Failed to connect to market data")


# === 4. Performance Report Generator ===

def generate_performance_report(days=7):
    """
    Generate a performance report from historical orders.
    """
    from projectx_connector import ProjectXClient
    from config import FUTURES
    import pandas as pd
    
    print(f"=== Performance Report (Last {days} Days) ===\n")
    
    # Initialize client
    px = ProjectXClient(
        FUTURES["topstep"]["username"],
        FUTURES["topstep"]["api_key"]
    )
    px.authenticate()
    
    # Get orders
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    orders = px.search_orders(start=start_time, end=end_time)
    
    if not orders:
        print("No orders found in the specified period.")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(orders)
    
    # Filter filled orders only
    filled_orders = df[df['status'] == 1]  # 1 = Filled
    
    if filled_orders.empty:
        print("No filled orders found.")
        return
    
    # Basic stats
    print(f"Total orders: {len(df)}")
    print(f"Filled orders: {len(filled_orders)}")
    print(f"Working orders: {len(df[df['status'] == 0])}")
    print(f"Cancelled orders: {len(df[df['status'] == 2])}")
    
    # Group by day
    filled_orders['date'] = pd.to_datetime(filled_orders['placedTimestamp']).dt.date
    daily_stats = filled_orders.groupby('date').agg({
        'id': 'count',
        'commission': 'sum'
    }).rename(columns={'id': 'trades', 'commission': 'total_commission'})
    
    print("\nDaily Summary:")
    print(daily_stats)
    
    # Calculate win rate (simplified - would need entry/exit matching)
    print("\nNote: For accurate P&L calculation, orders need to be matched as entry/exit pairs.")


# === 5. Telegram Test Script ===

def test_telegram_notifications():
    """
    Test Telegram notifications with various message types.
    """
    try:
        from telegram_notifier import TelegramNotifier
        
        print("=== Telegram Notification Test ===\n")
        
        # Load config
        if not os.path.exists('telegram_config.json'):
            print("❌ telegram_config.json not found!")
            print("Create it with: {\"bot_token\": \"YOUR_BOT_TOKEN\", \"chat_id\": \"YOUR_CHAT_ID\"}")
            return
        
        with open('telegram_config.json') as f:
            config = json.load(f)
        
        # Create notifier
        notifier = TelegramNotifier(config['bot_token'], config['chat_id'])
        notifier.start()
        
        ny_tz = ZoneInfo("America/New_York")
        
        print("Sending test messages...")
        
        # Test different message types
        time.sleep(1)
        notifier.send_message("🧪 Test 1: Basic message")
        print("✓ Sent basic message")
        
        time.sleep(2)
        notifier.send_trade_signal(
            side="BUY",
            price=20850.50,
            tp=20900.50,
            sl=20800.50,
            bar_time=datetime.now(ny_tz)
        )
        print("✓ Sent trade signal")
        
        time.sleep(2)
        notifier.send_order_filled("TP", 20900.50, 3)
        print("✓ Sent order fill notification")
        
        time.sleep(2)
        notifier.send_connection_status(False, "Market Data")
        print("✓ Sent connection status")
        
        time.sleep(2)
        notifier.send_error("Test error message", "System")
        print("✓ Sent error notification")
        
        time.sleep(2)
        stats = {
            'daily_pnl': 1250.00,
            'total_trades': 5,
            'win_rate': 60.0,
            'winning_trades': 3,
            'losing_trades': 2,
            'avg_win': 500.00,
            'avg_loss': 250.00,
            'best_trade': 750.00,
            'worst_trade': -300.00
        }
        notifier.send_daily_summary(stats)
        print("✓ Sent daily summary")
        
        time.sleep(2)
        notifier.send_market_update(20850.25, 20850.50, 20850.50)
        print("✓ Sent market update")
        
        # Wait for messages to send
        time.sleep(3)
        notifier.stop()
        
        print("\n✅ All test messages sent! Check your Telegram.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# === 6. Quick System Status ===

def show_system_status():
    """
    Display current system status without starting the full monitor.
    """
    from projectx_connector import ProjectXClient
    from config import FUTURES
    
    print("=== Quick System Status ===\n")
    
    try:
        # Initialize client
        px = ProjectXClient(
            FUTURES["topstep"]["username"],
            FUTURES["topstep"]["api_key"]
        )
        px.authenticate()
        
        # Get account info
        accounts = px._session.post(
            px.urls['accounts'],
            json={"onlyActiveAccounts": True},
            headers=px._headers(),
            timeout=px.request_timeout
        ).json().get("accounts", [])
        
        if accounts:
            account = accounts[0]
            print(f"Account: {account['name']}")
            print(f"Balance: ${account['balance']:,.2f}")
            print(f"Buying Power: ${account.get('buyingPower', account['balance']):,.2f}")
        
        # Check open orders
        open_orders = px.search_open_orders()
        print(f"\nOpen Orders: {len(open_orders)}")
        
        for order in open_orders[:5]:  # Show first 5
            order_id = order.get('id') or order.get('orderId')
            side = 'BUY' if order.get('side') == 0 else 'SELL'
            price = order.get('limitPrice') or order.get('stopPrice') or 'Market'
            print(f"  - {order_id}: {side} @ {price}")
        
        # Check recent fills
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)
        recent_orders = px.search_orders(start=start_time, end=end_time)
        
        filled = [o for o in recent_orders if o.get('status') == 1]
        print(f"\nFilled Orders (24h): {len(filled)}")
        
    except Exception as e:
        print(f"Error: {e}")


# === Main Menu ===

def main():
    """
    Main menu for helper utilities.
    """
    while True:
        print("\n=== Trading System Utilities ===")
        print("1. System Health Check")
        print("2. Test Telegram Notifications")
        print("3. Cancel All Open Orders")
        print("4. Quick System Status")
        print("5. Generate Performance Report")
        print("6. Test Market Data (30s)")
        print("0. Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == '1':
            check_system_health()
        elif choice == '2':
            test_telegram_notifications()
        elif choice == '3':
            cancel_all_open_orders()
        elif choice == '4':
            show_system_status()
        elif choice == '5':
            days = input("Number of days (default 7): ")
            days = int(days) if days else 7
            generate_performance_report(days)
        elif choice == '6':
            import asyncio
            asyncio.run(test_market_data(30))
        elif choice == '0':
            break
        else:
            print("Invalid option")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'health':
            check_system_health()
        elif sys.argv[1] == 'telegram':
            test_telegram_notifications()
        elif sys.argv[1] == 'cancel':
            cancel_all_open_orders()
        elif sys.argv[1] == 'status':
            show_system_status()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Available: health, telegram, cancel, status")
    else:
        main()