# Advanced Trading ML Model

A sophisticated machine learning-based trading system utilizing ensemble methods for market prediction and automated trading execution.

## 📊 Overview

This project implements an advanced trading system that combines multiple machine learning models to predict market movements and execute trades with robust risk management. The system uses ensemble learning techniques including Random Forest, XGBoost, and ElasticNet with a stacking approach.

## 🌟 Key Features

- **Ensemble Learning System**:
  - CatBoost Regressor
  - LightGBM Regressor
  - Stacking with CatBoost meta-learner

- **Advanced Model Optimization**:
  - Optuna-based hyperparameter optimization
  - Dynamic threshold adjustment
  - Cross-validation with time series split
  - Performance monitoring and adaptation

- **Risk Management**:
  - Dynamic stop-loss based on ATR
  - Trailing stop implementation
  - Position sizing rules
  - Session-based trading restrictions

## 📈 Technical Indicators

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
```bash
python trading-ai.ipynb or run using jupyter notebook
```

3. Configuration: Key parameters can be adjusted in the notebook:
`LOOKAHEAD`: Prediction timeframe [5, 15 minutes]
`SL_ATR_MULT`: Stop loss ATR multiplier
`TP_ATR_MULT`: Take profit ATR multiplier
`TRAIL_START_MULT`: Trailing stop trigger multiplier
`TRAIL_STOP_MULT`: Trailing stop distance multiplier
`BASE_CONTRACTS`: Minimum amount of contracts to use
`MAX_CONTRACTS`: Max amount of contracts to use based on confidence
`TRADE_THRESHOLD`: Threshold for entering trades

📋 Project Structure
```
Trading-ML-model/
│
├── live-trade/          # Python handler to use for trading with the model
├── models/              # Saved model files
├── source/              # Source code of training model
│     ├── data/         # Market data files
|     └── other/         # Market data files
│           └── trading-ai.ipynb     # Main training notebook
└── README.md            # Project documentation
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
