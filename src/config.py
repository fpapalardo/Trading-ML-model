# src/config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# this file is at project/src/config.py
ROOT = Path(__file__).resolve().parent.parent

API_USERNAME   = os.getenv("PROJECTX_USERNAME")
API_KEY        = os.getenv("PROJECTX_API_KEY")

DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
DB_DIR   = ROOT / "notebooks" / "dbs"
MODEL_DIR= ROOT / "models"
