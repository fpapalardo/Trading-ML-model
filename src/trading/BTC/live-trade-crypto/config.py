# config.py

# --- Exchange and API Settings ---
# Important: Use a separate file like .env for production or gitignored file
BINANCE_API_KEY = "3yC0OsSL2zJvkb4kBolLWU8gwoQNpTLjREdFSWSUlwCtG82Aru6mUmfN45SIuCko"
BINANCE_API_SECRET = "ITkcJf79GJhMJuzvpU88yRgqy5c0QThpA9ggjDuRZuIn5gUqRHE6PdXFOScfoeRG"

# --- Trading Mode ---
# Set to True for paper trading, False for live trading with real money
SIMULATED = True

# --- Trading Parameters ---
TRADING_SYMBOL = 'BTCUSDT'
TIMEFRAME = '5m'
LEVERAGE = 40.0

# Last Time 

# --- Position Sizing and Risk Management (from your backtest) ---
# This is used as the fraction of the balance to allocate to a position's margin
RISK_FRACTION = 1.0

# Stop Loss and Take Profit multipliers based on ATR
SL_ATR_MULT = 2.0
TP_ATR_MULT = 4.0

# --- Trailing Stop Parameters (from your backtest) ---
# Set TRAIL_START_MULT to 0.0 to disable trailing stops
TRAIL_START_MULT = 3.0#3.0  # When to start trailing (e.g., at 2.5 * ATR profit)
TRAIL_STOP_MULT = 1.0#1.0   # How far the trailing stop is from the peak price (e.g., 2.0 * ATR)

# --- Data Storage ---
DATA_FOLDER = "data"
CANDLE_DATA_FILE = f"{DATA_FOLDER}/{TRADING_SYMBOL.lower()}_{TIMEFRAME}_data.parquet"

# --- Feature Calculation ---
# This matches the ATR period from your backtest example `ATR_14_5min`
ATR_PERIOD = 14