# src/config.py

from pathlib import Path

# this file is at project/src/config.py
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
DB_DIR   = ROOT / "notebooks" / "dbs"
MODEL_DIR= ROOT / "models"
