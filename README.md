# Advanced Trading ML Model (Outdated)
## 🛠 Prerequisites

- python 3.12.10

⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/Trading-ML-model.git
cd Trading-ML-model
```

Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
or for windows use https://pypi.org/project/pyenv-win/

**Tested and ran with python 3.12.10**

Install dependencies:
```bash
pip install -r requirements.txt
```

🚀 Usage
1. Data Preparation:
    - Place your market data in the /data/raw directory
    - Supported format: CSV with OHLCV data

2. Model Training:
Run using jupyter notebook for experimentation there is helper functions to reduce the amount of cells in each notebooks

📋 Project Structure
```
TRADING-ML-MODEL/
├── .env                       # Environment variable definitions (e.g. API keys)
├── .gitattributes             # Git attributes for handling line endings, etc.
├── .gitignore                 # Files and dirs to exclude from Git
├── CONTRIBUTING.md            # Guidelines for contributing to this repo
├── ISSUE_TEMPLATE.md          # Template for new issue submissions
├── PULL_REQUEST_TEMPLATE.md   # Template for pull request descriptions
├── README.md                  # Project overview and setup instructions
├── requirements.txt           # Pin-exact Python dependencies
└── setup.py                   # Package installation script

data/
├── live                      # Incoming live data ingestion
├── processed                 # Cleaned & transformed datasets
└── raw                       # Raw/unmodified data dumps

docs/
├── model_implementation_details.md   # In-depth notes on model code
├── model_recommendations.md          # Detailed model suggestions
├── model_recommendations_summary.md  # High-level summary of recommendations
└── tasks.md                          # Project to-do list and milestones

models/
└── rf_classifier/            # Random Forest classifier artifacts
    └── v1/                   # Version 1 model outputs

notebooks/
├── dbs/                      # Database exploration notebooks
├── parquet/                  # Parquet validation & utils
│   ├── validate_data.py      # Script to check parquet integrity
│   ├── BTC/                  # Bitcoin data notebooks
│   └── NQ/                   # Nasdaq data notebooks
├── pkl/                      # Pickle serialization demos
├── reporting/                # Reporting & visualization notebooks
└── research/                 # Research experiments
    ├── notebooks_metadata/   # Metadata for research notebooks
    └── NQ/                   # Nasdaq-focused studies

outputs/
├── backtests/
│   ├── BTC/                  # Bitcoin backtest results
│   └── NQ/                   # Nasdaq backtest results
├── metrics/                  # Generated performance metrics
└── trading_logs/             # Logs from trading runs

src/
├── config.py                 # Global configuration settings
├── exploration/              # Ad-hoc exploratory code
│   ├── feature-iterator/     # Iterating feature-set experiments
│   ├── LLMs/                 # Large Language Model trials
│   │   └── parquet/          # Parquet I/O for LLM data
│   └── logs/                 # Exploration log files
├── features/                 # Feature engineering modules
│   ├── composite.py          # Composite feature builders
│   ├── indicators.py         # Standard technical indicators
│   ├── momentum.py           # Momentum feature logic
│   ├── price_action.py       # Price-action feature logic
│   ├── registry.py           # Central feature registry
│   ├── session.py            # Trading session time calculations
│   ├── trend.py              # Trend indicator functions
│   ├── volatility.py         # Volatility metrics
│   ├── volume.py             # Volume-based features
│   ├── volume_stats.py       # Volume statistics utilities
│   └── __init__.py           # Package init
├── models/                   # Model definitions & serializers
│   ├── classifier/           # Classification model code
│   └── regression/           # Regression model code
├── optimization/             # Hyperparameter tuning scripts
│   ├── classification.py     # Classification tuning logic
│   ├── common.py             # Shared optimization utilities
│   ├── regression.py         # Regression tuning logic
│   └── __init__.py
├── pipeline/                 # Data pipeline orchestration
├── trading/                  # Live trading logic
│   ├── BTC/                  # Bitcoin trading scripts
│   │   └── live-trade-crypto/
│   │       ├── indicator_calculation.py  # Crypto indicator funcs
│   │       └── live_trading.py           # Crypto live execution
│   └── NQ/                   # Nasdaq trading scripts
│       ├── helper_scripts.py
│       ├── indicator_calculation.py      # Indicator funcs for NQ
│       ├── live_trading.py               # Live trading loop
│       ├── order_manager.py              # Order placement & tracking
│       ├── projectx_connector.py         # Project X API client
│       ├── signalr_market_hub.py         # Market data hub handler
│       ├── signalr_user_hub.py           # User hub handler
│       ├── startup.py                    # Initialization routines
│       ├── telegram_config.json          # Telegram notifier config
│       ├── telegram_notifier.py          # Alerts via Telegram
│       ├── trading_monitor_v2.py         # Monitoring/trading orchestrator
│       ├── logs/                         # Trading session logs
│       └── model/                        # On-the-fly model artifacts
└── utils/                    # General helper scripts
    ├── backtest.py
    ├── crypto_refactorme_cleaner.py
    ├── data_cleaner.py
    ├── data_loader.py
    ├── helpers.py
    ├── indeces_market_checker.py
    ├── labeling_utils.py
    ├── metrics.py
    ├── pipeline.py
    ├── training_live_validator_refactorme.py
    └── __init__.py

tests/                       # Unit & integration tests

```

🔬 **Model Architecture**
- **Feature Engineering**  
  Multi-timeframe technical indicators (EMA, MACD, ATR, VWAP, Choppiness Index), volume & price-action metrics  
- **Stacked Ensemble**  
  Regression & classification base learners (RF, XGB, LGBM) plus a meta-learner for final signal  
- **Adaptive Hyperopt**  
  Scheduled Optuna tuning to recalibrate hyperparameters based on recent market data  
- **Risk Controls & Sizing**  
  ATR-based stop-loss, dynamic take-profit, position scaling in/out, OCO orders

📊 **Performance Monitoring**
- **Win Rate** (long vs. short)  
- **Profit Factor** (gross P&L / gross loss)  
- **Sharpe & Sortino Ratios**  
- **MFE/MAE Analysis** (max favorable/adverse excursion)  
- **Trade Duration & Slippage Tracking**  
- **Signal Confidence Threshold** Tuning

⚠️ **Disclaimer**  
This software is for **educational purposes only**. Live trading involves significant risk – never deploy capital you can’t afford to lose. Authors assume **no liability** for any trading outcomes.

📝 **License**  
This project is licensed under the [MIT License](LICENSE).

🤝 **Contributing**  
Contributions, bug reports, and feature requests are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

📧 **Contact**  
- **Name:** Franco Papalardo
- **GitHub:** [fpapalardo](https://github.com/fpapalardo)
