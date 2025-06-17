# src/config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# this file is at project/src/config.py
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

FUTURES = {
    "topstep": {
        "username": os.getenv("FUTURES_TOPSTEP_USERNAME"),
        "api_key":  os.getenv("FUTURES_TOPSTEP_API_KEY"),
    }
}

CRYPTO = {
    "binance": {
        "api_key": os.getenv("CRYPTO_BINANCE_API_KEY"),
        "secret":  os.getenv("CRYPTO_BINANCE_SECRET"),
    }
}

DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
DB_DIR   = ROOT / "notebooks" / "dbs"
MODEL_DIR= ROOT / "models"
