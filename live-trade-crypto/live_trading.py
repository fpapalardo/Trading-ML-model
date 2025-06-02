import time
import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

# === Load API Keys ===
api_key = "toC62A7o8kTd6dzqFCkSbr0tKgqALo8ZCmsnwVr4OrgSYrLtPPQ9LKa9TIySA9Ph"
api_secret = "sOr2jBFsBeKTEm0T5vBQwzogDBSIKicOgMv3syCf3THhgFM1JPVqZTH5YWqF4rWB"
client = Client(api_key, api_secret)

# === Get Balance ===
def get_margin_balance(asset="USDT", is_isolated=True, symbol="BTCUSDT"):
    try:
        if is_isolated:
            res = client.get_isolated_margin_account()
            assets = res["assets"]
            for a in assets:
                if a["symbol"] == symbol:
                    for_balance = a["quoteAsset"] if asset == "USDT" else a["baseAsset"]
                    available = float(for_balance["free"])
                    borrowed = float(for_balance["borrowed"])
                    return available, borrowed
        else:
            res = client.get_margin_account()
            for bal in res["userAssets"]:
                if bal["asset"] == asset:
                    return float(bal["free"]), float(bal["borrowed"])
    except Exception as e:
        print(f"❌ Error fetching balance: {e}")
        return 0, 0

# === Borrow Margin Funds ===
def borrow_margin(asset="USDT", amount=50, symbol="BTCUSDT", is_isolated=True):
    try:
        client.create_margin_loan(
            asset=asset,
            amount=str(amount),
            isIsolated=is_isolated,
            symbol=symbol
        )
        print(f"✅ Borrowed {amount} {asset}")
        return True
    except BinanceAPIException as e:
        print(f"❌ Borrow failed: {e}")
        return False

# === Place Market Order ===
def place_margin_order(symbol="BTCUSDT", side="BUY", quantity=0.001, is_isolated=True):
    try:
        order = client.create_margin_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity,
            isIsolated=is_isolated
        )
        print(f"✅ {side} order executed: {order['executedQty']} {symbol.replace('USDT','')}")
    except BinanceAPIException as e:
        print(f"❌ Order failed: {e}")

# === Execute Auto 2x Leverage Trade ===
def execute_2x_leverage_trade(symbol="BTCUSDT", is_isolated=True):
    quote_asset = "USDT"

    # Step 1: Get available balance
    owned, borrowed = get_margin_balance(asset=quote_asset, is_isolated=is_isolated, symbol=symbol)
    print(f"💰 Owned: {owned} {quote_asset} | Borrowed: {borrowed} {quote_asset}")

    if owned < 10:
        print("❌ Not enough owned balance to open leveraged trade.")
        return

    borrow_amt = owned  # For 2x
    total_trade_capital = owned + borrow_amt
    print(f"📊 Total capital for 2x: {total_trade_capital} USDT")

    # Step 2: Borrow funds
    success = borrow_margin(asset=quote_asset, amount=borrow_amt, symbol=symbol, is_isolated=is_isolated)
    if not success:
        return

    time.sleep(2)  # Wait for borrow confirmation

    # Step 3: Market Buy
    price = float(client.get_symbol_ticker(symbol=symbol)["price"])
    quantity = round(total_trade_capital / price, 6)
    print(f"📈 Market BUY {quantity} {symbol.replace('USDT','')} @ ${price:.2f}")
    place_margin_order(symbol=symbol, side="BUY", quantity=quantity, is_isolated=is_isolated)

# === Run the script ===
if __name__ == "__main__":
    execute_2x_leverage_trade("BTCUSDT")
