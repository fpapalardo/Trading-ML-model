# Advanced Trading ML Model
# (OUTDATED)
## 🛠 Prerequisites

```bash
python 3.12.10
pandas
numpy
scikit-learn
seaborn
lightgbm
catboost
optuna
ta
matplotlib
joblib
shap
```

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
    - Place your market data in the /data directory
    - Supported format: CSV with OHLCV data

2. Model Training:
Run using jupyter notebook for experimentation there is helper functions to reduce the amount of cells in each notebooks

📋 Project Structure
```
Trading-ML-model/
│
├── live-trade/          # Python handler to use for trading with the model
│     ├── trading/                  # Files for NT8
|     ├── indicator_calculation.py  # Indicator processing
|     └── live_trading.py           # Live Trading module
├── src/              # Source code of training model
│     ├── data/         # Market data files
│     ├── notebooks/        # Market data files
│     │      ├── dbs/       # Optuna DBs
│     │      ├── parquet/   # Parquet Files
│     │      ├── pkl/       # PKL Files
│     │      └── *.ipynb    # Model Training
|     ├── data_loader.py    # Data loader for initialization
|     ├── backtest.py       # Backtest Logic
|     ├── helpers.py        # Helper functions
|     └── labeling_utils.py # Labeling
├── README.md            # Project documentation
├── .gitignore
├── .gittributes
└── requirements.txt
```

🔬 Model Architecture
- Feature Engineering: Custom technical indicators and market metrics
- Ensemble Approach: Combines predictions from multiple models
- Dynamic Optimization: Continuous parameter adjustment based on market conditions
- Risk Management: Integrated position sizing and stop management

📊 Performance Monitoring
The system includes:

- Win rate tracking
- Profit factor calculation
- Sharpe ratio monitoring
- Trade duration analysis
- Dynamic threshold adjustment

⚠️ Disclaimer
This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

📝 License


🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.

📧 Contact
-

** Numpy version was recently updated to latest which was causing not being able to use a higher python version although no higher version was tested yet
