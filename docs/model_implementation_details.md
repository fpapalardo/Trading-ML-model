# Model Implementation Details for NQ Futures Trading

This document provides detailed explanations and implementation guidance for the model recommendations outlined in `model_recommendations.md`.

## Understanding NQ Futures Characteristics

The Nasdaq-100 futures (NQ) have specific characteristics that influence model selection:

1. **High Volatility**: NQ typically has higher volatility than other index futures like ES (S&P 500)
2. **Tech Sector Dominance**: Heavily influenced by technology stocks
3. **Gap Trading**: Often experiences significant overnight gaps
4. **Extended Trading Hours**: Active during both US and Asian sessions
5. **News Sensitivity**: Highly reactive to tech earnings and economic data

## Detailed Model Implementation

### 1. Tree-Based Models Implementation

#### CatBoost with Quantile Loss

```python
from catboost import CatBoostRegressor

# For capturing upside potential (biased toward positive moves)
catboost_up = CatBoostRegressor(
    loss_function='Quantile:alpha=0.7',  # Bias toward upper quantile
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_strength=1.0,
    bagging_temperature=1.0,
    has_time=True,  # Important for time series
    verbose=0
)

# For capturing downside risk (biased toward negative moves)
catboost_down = CatBoostRegressor(
    loss_function='Quantile:alpha=0.3',  # Bias toward lower quantile
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_strength=1.0,
    bagging_temperature=1.0,
    has_time=True,
    verbose=0
)
```

**Why this works for NQ**: The asymmetric loss functions help capture the skewed distribution of returns in NQ futures, which often exhibit sharp upward movements followed by slower downward corrections.

#### LightGBM with Different Objectives

```python
import lightgbm as lgb

# For general prediction
lgbm_huber = lgb.LGBMRegressor(
    objective='huber',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    importance_type='gain'
)

# For volatility prediction
lgbm_rmse = lgb.LGBMRegressor(
    objective='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    importance_type='gain'
)
```

**Why this works for NQ**: LightGBM's efficiency allows for faster training and iteration, which is valuable when dealing with the rapidly changing patterns in NQ futures.

### 2. Deep Learning Models Implementation

#### LSTM for Sequential Patterns

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, RegressorMixin

class LSTMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, units=64, dropout=0.2, learning_rate=0.001):
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self):
        model = Sequential([
            LSTM(self.units, input_shape=(self.input_shape, 1), return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(self.dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def fit(self, X, y, **kwargs):
        # Reshape X for LSTM: [samples, timesteps, features]
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model()
        self.model.fit(
            X_reshaped, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        return self
    
    def predict(self, X):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X_reshaped).flatten()
```

**Why this works for NQ**: LSTM models can capture the sequential patterns in NQ futures, particularly the momentum effects that persist over multiple time periods. They're especially effective at modeling the overnight session transitions.

#### 1D CNN for Pattern Recognition

```python
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D

class CNN1DWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, filters=64, kernel_size=3, dropout=0.2, learning_rate=0.001):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self):
        model = Sequential([
            Conv1D(self.filters, kernel_size=self.kernel_size, activation='relu', 
                   input_shape=(self.input_shape, 1)),
            Dropout(self.dropout),
            Conv1D(self.filters*2, kernel_size=self.kernel_size, activation='relu'),
            Dropout(self.dropout),
            GlobalAveragePooling1D(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def fit(self, X, y, **kwargs):
        # Reshape X for CNN: [samples, features, channels]
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model()
        self.model.fit(
            X_reshaped, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        return self
    
    def predict(self, X):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X_reshaped).flatten()
```

**Why this works for NQ**: CNNs excel at identifying local patterns in the data, such as chart patterns and indicator crossovers, which are particularly relevant for NQ futures trading.

### 3. Advanced Stacking Implementation

#### Two-Tier Stacking

```python
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV

# Tier 1: Same-type model stacking
catboost_models = [
    ('catboost_quantile_0.6', CatBoostRegressor(loss_function='Quantile:alpha=0.6')),
    ('catboost_quantile_0.7', CatBoostRegressor(loss_function='Quantile:alpha=0.7')),
    ('catboost_mae', CatBoostRegressor(loss_function='MAE'))
]

lightgbm_models = [
    ('lgbm_huber', lgb.LGBMRegressor(objective='huber')),
    ('lgbm_quantile', lgb.LGBMRegressor(objective='quantile', alpha=0.6)),
    ('lgbm_rmse', lgb.LGBMRegressor(objective='rmse'))
]

# Tier 1 stacks
catboost_stack = StackingRegressor(
    estimators=catboost_models,
    final_estimator=RidgeCV(),
    cv=TimeSeriesSplit(n_splits=3)
)

lightgbm_stack = StackingRegressor(
    estimators=lightgbm_models,
    final_estimator=RidgeCV(),
    cv=TimeSeriesSplit(n_splits=3)
)

# Tier 2: Heterogeneous model stacking
final_stack = StackingRegressor(
    estimators=[
        ('catboost_ensemble', catboost_stack),
        ('lightgbm_ensemble', lightgbm_stack),
        ('lstm', LSTMWrapper(input_shape=X_train.shape[1])),
        ('cnn', CNN1DWrapper(input_shape=X_train.shape[1]))
    ],
    final_estimator=ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                eps=1e-3,
                                n_alphas=100,
                                max_iter=1000),
    cv=TimeSeriesSplit(n_splits=5),
    passthrough=True  # Include original features
)
```

**Why this works for NQ**: The two-tier stacking approach allows for specialized models at the first tier (e.g., models that excel at capturing different aspects of NQ price movement) while the second tier combines these specialized predictions into a more robust overall forecast.

### 4. Market Regime Classification

```python
def identify_market_regime(df, window=20):
    """
    Classify market regimes based on volatility and trend
    
    Returns:
    - 0: Low volatility, range-bound
    - 1: Low volatility, trending
    - 2: High volatility, range-bound
    - 3: High volatility, trending
    """
    # Calculate volatility (ATR relative to its moving average)
    vol_ratio = df['ATR_14_5min'] / df['ATR_14_5min'].rolling(window*5).mean()
    
    # Calculate trend strength
    trend_strength = abs(df['EMA_20_5min'] - df['EMA_50_5min']) / df['ATR_14_5min']
    
    # Classify regimes
    high_vol = vol_ratio > 1.2
    strong_trend = trend_strength > 1.0
    
    regimes = pd.Series(0, index=df.index)  # Default: low vol, range-bound
    regimes[high_vol & ~strong_trend] = 2    # High vol, range-bound
    regimes[~high_vol & strong_trend] = 1    # Low vol, trending
    regimes[high_vol & strong_trend] = 3     # High vol, trending
    
    return regimes

# Train regime-specific models
def train_regime_models(X, y, regimes):
    models = {}
    for regime in range(4):
        mask = (regimes == regime)
        if sum(mask) > 100:  # Ensure enough samples
            X_regime = X[mask]
            y_regime = y[mask]
            
            model = CatBoostRegressor(iterations=500, verbose=0)
            model.fit(X_regime, y_regime)
            models[regime] = model
    
    return models

# Predict using regime-specific models
def predict_with_regime_models(X, regimes, models, default_model):
    predictions = np.zeros(len(X))
    
    for regime in range(4):
        mask = (regimes == regime)
        if regime in models and sum(mask) > 0:
            predictions[mask] = models[regime].predict(X[mask])
        elif sum(mask) > 0:
            predictions[mask] = default_model.predict(X[mask])
    
    return predictions
```

**Why this works for NQ**: NQ futures exhibit distinct behavior in different market regimes. For example, during high volatility trending markets (often seen during tech earnings season), aggressive trend-following strategies work well. In contrast, during low volatility range-bound conditions, mean-reversion strategies are more effective.

## NQ-Specific Feature Engineering

### 1. Overnight Gap Features

```python
def add_overnight_gap_features(df):
    """Add features related to overnight price gaps"""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Create session date
    df['session_date'] = df.index.date
    
    # Get last price of previous session and first price of current session
    df['prev_session_close'] = df.groupby('session_date')['close'].transform(lambda x: x.iloc[-1].shift(1))
    df['curr_session_open'] = df.groupby('session_date')['open'].transform('first')
    
    # Calculate overnight gap
    df['overnight_gap_pct'] = (df['curr_session_open'] - df['prev_session_close']) / df['prev_session_close']
    
    # Gap features
    df['gap_up'] = (df['overnight_gap_pct'] > 0.001).astype(int)
    df['gap_down'] = (df['overnight_gap_pct'] < -0.001).astype(int)
    df['gap_magnitude'] = df['overnight_gap_pct'].abs()
    
    # Gap fill features
    df['gap_fill_price'] = np.where(
        df['gap_up'], df['prev_session_close'], 
        np.where(df['gap_down'], df['prev_session_close'], np.nan)
    )
    
    # Distance to gap fill
    df['distance_to_gap_fill'] = np.where(
        ~df['gap_fill_price'].isna(),
        (df['gap_fill_price'] - df['close']) / df['close'],
        0
    )
    
    return df
```

### 2. Tech Sector Correlation Features

```python
def add_tech_correlation_features(df, tech_data):
    """
    Add correlation features with tech sector ETF (QQQ) and major tech stocks
    
    Parameters:
    - df: DataFrame with NQ data
    - tech_data: DataFrame with QQQ and tech stock data aligned to the same timeframe
    """
    # Calculate rolling correlations
    for symbol in ['QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']:
        if symbol in tech_data.columns:
            # 1-hour rolling correlation
            df[f'{symbol}_corr_1h'] = df['close'].rolling(12).corr(tech_data[symbol])
            
            # 4-hour rolling correlation
            df[f'{symbol}_corr_4h'] = df['close'].rolling(48).corr(tech_data[symbol])
            
            # Relative strength
            df[f'{symbol}_rel_strength'] = (
                df['close'].pct_change(12) - tech_data[symbol].pct_change(12)
            )
    
    # Tech sector breadth
    if all(s in tech_data.columns for s in ['AAPL', 'MSFT', 'AMZN', 'GOOGL']):
        # Count how many tech stocks are up in the last hour
        tech_up_count = sum(
            tech_data[s].pct_change(12) > 0 
            for s in ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
        )
        df['tech_breadth'] = tech_up_count / 4  # Normalize to 0-1
    
    return df
```

### 3. Volume Profile Features

```python
def add_volume_profile_features(df, lookback=20):
    """Add volume profile features to identify support/resistance levels"""
    # Create price bins (rounded to nearest 5 points for NQ)
    df['price_bin'] = (df['close'] / 5).round() * 5
    
    # Rolling volume profile
    for i in range(lookback):
        if i+1 < len(df):
            # Get previous N days
            hist_data = df.iloc[max(0, i-lookback):i+1]
            
            # Calculate volume profile
            vol_profile = hist_data.groupby('price_bin')['volume'].sum()
            
            # Find high volume nodes (potential support/resistance)
            high_vol_nodes = vol_profile[vol_profile > vol_profile.quantile(0.8)].index.tolist()
            
            # Calculate distance to nearest high volume node
            if high_vol_nodes:
                df.loc[i, 'dist_to_vol_node'] = min(
                    abs(df.loc[i, 'close'] - node) for node in high_vol_nodes
                ) / df.loc[i, 'ATR_14_5min']
    
    # Value Area features
    df['in_value_area'] = 0
    
    # Calculate Value Area (70% of volume)
    for date in df['session_date'].unique():
        day_data = df[df['session_date'] == date]
        vol_profile = day_data.groupby('price_bin')['volume'].sum()
        
        # Sort by volume
        sorted_profile = vol_profile.sort_values(ascending=False)
        
        # Get 70% of volume
        cumsum = sorted_profile.cumsum() / sorted_profile.sum()
        value_area = sorted_profile[cumsum <= 0.7].index.tolist()
        
        # Mark if price is in value area
        if value_area:
            min_va = min(value_area)
            max_va = max(value_area)
            in_va = (day_data['close'] >= min_va) & (day_data['close'] <= max_va)
            df.loc[day_data.index, 'in_value_area'] = in_va.astype(int)
    
    return df
```

## Combining Models for Optimal Performance

The key to successful NQ futures trading is not just having sophisticated models, but combining them in ways that leverage their strengths for different market conditions:

### Ensemble Weighting Strategy

```python
def dynamic_ensemble_weights(X, models, market_features):
    """
    Dynamically adjust model weights based on market conditions
    
    Parameters:
    - X: Features for prediction
    - models: Dictionary of trained models
    - market_features: DataFrame with market regime indicators
    
    Returns:
    - Weighted prediction
    """
    # Base weights (determined through validation)
    base_weights = {
        'catboost_quantile': 0.3,
        'catboost_mae': 0.2,
        'lightgbm_huber': 0.2,
        'xgboost_reg': 0.15,
        'lstm_reg': 0.15
    }
    
    # Adjust weights based on market conditions
    adjusted_weights = base_weights.copy()
    
    # In high volatility, increase weight of robust models
    if market_features['vol_regime'] == 'high':
        adjusted_weights['catboost_mae'] += 0.1
        adjusted_weights['lightgbm_huber'] += 0.05
        adjusted_weights['catboost_quantile'] -= 0.1
        adjusted_weights['lstm_reg'] -= 0.05
    
    # In strong trends, increase weight of trend-following models
    if market_features['trend_strength'] > 1.5:
        adjusted_weights['xgboost_reg'] += 0.1
        adjusted_weights['lstm_reg'] += 0.05
        adjusted_weights['catboost_mae'] -= 0.1
        adjusted_weights['lightgbm_huber'] -= 0.05
    
    # Normalize weights to sum to 1
    total = sum(adjusted_weights.values())
    adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
    
    # Make predictions with each model
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X)
    
    # Combine predictions using adjusted weights
    weighted_pred = sum(predictions[name] * weight for name, weight in adjusted_weights.items())
    
    return weighted_pred
```

By implementing these detailed recommendations, the trading system will be better equipped to handle the unique characteristics of NQ futures trading while maintaining compatibility with the existing codebase.