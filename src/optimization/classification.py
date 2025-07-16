import numpy as np
import optuna
import json

import pandas as pd

from optimization.common import auto_ts_split, create_study, get_storage_uri, blend_scores
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_sample_weight, compute_class_weight
from sklearn.metrics import f1_score, make_scorer, fbeta_score
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from config import DB_DIR
from utils.backtest import evaluate_classification, evaluate_crypto_classification

def tune_xgboost(
    X_train, y_train,
    market: str,
    n_trials: int = 50,
    unique_id: str = None,
    n_jobs: int = -1
) -> dict:
    """
    Optimize an XGBClassifier using weighted CV and training-set F1 (80/20).
    """
    study_name = f"xgb_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train)),
            'eval_metric': 'logloss'
        }
        tscv = auto_ts_split(len(y_train))
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            weights = compute_sample_weight(class_weight='balanced', y=y_tr)
            model = XGBClassifier(**params, random_state=42, n_jobs=n_jobs)
            model.fit(X_tr, y_tr, sample_weight=weights)
            preds = model.predict(X_val)
            cv_scores.append(f1_score(y_val, preds, average='macro'))
        cv_mean = np.mean(cv_scores)
        final = XGBClassifier(**params, random_state=42, n_jobs=n_jobs)
        final.fit(X_train, y_train)
        train_f1 = f1_score(y_train, final.predict(X_train), average='macro')
        score = 0.8 * cv_mean + 0.2 * train_f1
        trial.set_user_attr('cv_mean', cv_mean)
        trial.set_user_attr('train_f1', train_f1)
        return score

    study = create_study(
        study_name=study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_lgbm(
    X_train, y_train,
    market: str,
    n_trials: int = 50,
    unique_id: str = None,
    n_jobs: int = -1
) -> dict:
    """
    Optimize an LGBMClassifier using weighted CV and training-set F1 (80/20).
    """
    study_name = f"lgbm_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'random_state': 42,
            'n_jobs': n_jobs,
            'verbosity': -1
        }
        tscv = auto_ts_split(len(y_train))
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            weights = compute_sample_weight(class_weight='balanced', y=y_tr)
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=weights)
            preds = model.predict(X_val)
            cv_scores.append(f1_score(y_val, preds, average='macro'))
        cv_mean = np.mean(cv_scores)
        final = lgb.LGBMClassifier(**params)
        final.fit(X_train, y_train)
        train_f1 = f1_score(y_train, final.predict(X_train), average='macro')
        score = 0.8 * cv_mean + 0.2 * train_f1
        trial.set_user_attr('cv_mean', cv_mean)
        trial.set_user_attr('train_f1', train_f1)
        return score

    study = create_study(
        study_name=study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_rf(
    X_train, y_train,
    labeled,
    market: str,
    n_trials: int = 50,
    unique_id: str = None,
    n_jobs: int = -1
) -> dict:
    """
    Optimize RandomForestClassifier using cross-val + OOB blend.
    """
    study_name = f"rf_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"

    if not isinstance(labeled.index, pd.DatetimeIndex):
        if "datetime" in labeled.columns:
            labeled = labeled.set_index("datetime")
        else:
            raise ValueError("No 'datetime' column to set as index!")
    if labeled.index.tz is None:
        labeled = labeled.tz_localize("UTC")
    labeled = labeled.tz_convert("America/New_York")
    # fbeta_scorer = make_scorer(
    #     fbeta_score,
    #     beta=2,               # example: β=2 emphasizes recall twice as much as precision
    #     average='binary'      # or 'macro' if you have multiple classes
    # )

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 5000, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 50, 1000, step=50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 200, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.1, 0.2, 0.5, 0.8, 1.0]),
            'bootstrap': True,
            'oob_score': False,
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
        # F1 --->
        # tscv = auto_ts_split(len(y_train))
        # model = RandomForestClassifier(**params, random_state=42, n_jobs=n_jobs)
        # cv_scores = cross_val_score(
        #     model, X_train, y_train,
        #     cv=tscv, scoring='f1_macro', n_jobs=n_jobs
        # )
        # model.fit(X_train, y_train)
        # oob = model.oob_score_
        # score = 0.8 * cv_scores.mean() + 0.2 * oob
        # trial.set_user_attr('cv_mean', cv_scores.mean())
        # trial.set_user_attr('oob_score', oob)
        # return score
        # F1 <---

        # Gemini --->
        tscv = auto_ts_split(len(y_train))
    
        fold_scores = []
        fold_profit_factors = []
        fold_num_trades = []

        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(**params, random_state=42, n_jobs=n_jobs)
            model.fit(X_train_fold, y_train_fold)

            # predictions = model.predict_proba(X_val_fold)
            predictions   = model.predict(X_val_fold)
            # turn [0,1,2]-labels into a one-hot array of shape (n,3)
            backtest_results = evaluate_classification(X_val_fold, 
                                                        predictions, 
                                                        labeled, {}, TRAIL_START_MULT=0,
                                                        TRAIL_STOP_MULT=0,
                                                        TICK_VALUE=6)
            
            profit_factor = backtest_results.get('profit_factor', 0)
            num_trades = backtest_results.get('trades', 0)
            print(f"Profit factor this fold {profit_factor}, with {num_trades} trades")

            if num_trades == 0:
                fold_scores.append(0.0)
                continue

            # Store metrics for later analysis
            fold_profit_factors.append(profit_factor)
            fold_num_trades.append(num_trades)

            # Handle edge case: profit_factor can be infinite if there are no losing trades.
            # This is great, but can break the optimizer. We cap it at a high value.
            if np.isinf(profit_factor):
                profit_factor = 100 # A high, but finite number

            # Calculate the score for this fold
            score = profit_factor * np.log1p(num_trades)
            fold_scores.append(score)

        score = np.mean(fold_scores)

        trial.set_user_attr('mean_profit_factor', np.mean(fold_profit_factors))
        trial.set_user_attr('mean_num_trades', np.mean(fold_num_trades))

        return score
        # Gemini <---

        # # Pareto --->
        # model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        # # 2. run cross‐validation, capturing both precision (win‐rate) and recall (trade rate)
        # scoring = {
        #     'precision': 'precision_macro',  
        #     'recall':    'recall_macro'      
        # }
        # cv_results = cross_validate(
        #     model, X_train, y_train,
        #     cv=auto_ts_split(len(y_train)),
        #     scoring=scoring,
        #     n_jobs=-1
        # )

        # precision_mean = cv_results['test_precision'].mean()
        # recall_mean    = cv_results['test_recall'].mean()

        # # 3. record for later inspection
        # trial.set_user_attr('precision', precision_mean)
        # trial.set_user_attr('recall',    recall_mean)

        # # 4. return a tuple of two objectives
        # return precision_mean, recall_mean
        # # Pareto <---
        #FB 
        # model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        # # CV on Fβ‐score directly
        # cv_scores = cross_val_score(
        #     model, X_train, y_train,
        #     cv=auto_ts_split(len(y_train)),
        #     scoring=fbeta_scorer,
        #     n_jobs=-1,
        #     error_score=0.0
        # )

        # # Optionally combine with OOB if you like:
        # model.fit(X_train, y_train)
        # oob = model.oob_score_

        # # e.g. weight CV-Fβ 80% and OOB 20%
        # combined = 0.8 * cv_scores.mean() + 0.2 * oob

        # trial.set_user_attr('cv_fbeta_mean', cv_scores.mean())
        # trial.set_user_attr('oob_score',      oob)
        # return combined

    study = create_study(
        study_name=study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_rf_grok(
    X_train, y_train,
    labeled,
    market: str,
    n_trials: int = 50,
    unique_id: str = None,
    n_jobs: int = -1
) -> dict:
    """
    Optimize RandomForestClassifier using cross-val + OOB blend.
    """
    study_name = f"rf_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"

    if not isinstance(labeled.index, pd.DatetimeIndex):
        if "datetime" in labeled.columns:
            labeled = labeled.set_index("datetime")
        else:
            raise ValueError("No 'datetime' column to set as index!")
    if labeled.index.tz is None:
        labeled = labeled.tz_localize("UTC")
    labeled = labeled.tz_convert("America/New_York")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 5000, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 50, 1000, step=50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 200, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.1, 0.2, 0.5, 0.8, 1.0]),
            'bootstrap': True,
            'oob_score': False,
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }

        tscv = auto_ts_split(len(y_train))
    
        fold_total_pnls = []

        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(**params, random_state=42, n_jobs=n_jobs)
            model.fit(X_train_fold, y_train_fold)

            predictions = model.predict(X_val_fold)
            backtest_results = evaluate_classification(X_val_fold, 
                                                       predictions, 
                                                       labeled, {}, TRAIL_START_MULT=0,
                                                       TRAIL_STOP_MULT=0,
                                                       TICK_VALUE=6)
            
            results = backtest_results.get('results', [])
            # 2. Normalize into a list of dicts
            if isinstance(results, pd.DataFrame):
                records = results.to_dict('records')
            else:
                records = results
            

            total_pnl = sum(trade.get('pnl', 0) for trade in records if isinstance(trade, dict))
            fold_total_pnls.append(total_pnl)

        score = np.mean(fold_total_pnls)
        return score

    study = create_study(
        study_name=study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

import numpy as np

def calculate_max_drawdown(equity_curve, starting_capital=2000.0):
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    drawdowns   = (running_max - eq) / starting_capital
    return float(np.max(drawdowns))

def max_consecutive_loss(trade_pnls: list) -> float:
    run_sum  = 0.0
    max_loss = 0.0
    for pnl in trade_pnls:
        run_sum += pnl
        if run_sum > 0:
            run_sum = 0.0
        max_loss = max(max_loss, -run_sum)
    return float(max_loss)

def calculate_custom_score(trades_data: list,
                           starting_capital: float = 2000.0
                          ) -> tuple:
    """
    Returns:
      score,
      total_pnl,
      num_trades,
      mdd                # fraction of start cap
      mcl                # absolute max consecutive loss
      mcl_ratio          # mcl / start cap
    """
    if not trades_data:
        return -1e9, 0.0, 0, 0.0, 0.0, 0.0

    pnls = [t.get('pnl', 0.0) for t in trades_data if isinstance(t, dict)]
    total_pnl  = float(np.sum(pnls))
    num_trades = len(pnls)
    equity_curve = starting_capital + np.cumsum(pnls)
    mdd    = calculate_max_drawdown(equity_curve, starting_capital)
    mcl    = max_consecutive_loss(pnls)
    mcl_ratio = mcl / starting_capital if starting_capital else 0.0

    score = (total_pnl * np.log1p(num_trades)
             / (1 + mdd)
             / (1 + mcl_ratio))
    return score, total_pnl, num_trades, mdd, mcl, mcl_ratio


def calculate_custom_score_crypto(records, initial_balance=None):
    """
    Calculate custom score with proportional thresholds for crypto trading.
    
    Parameters:
    - records: list of trade records
    - initial_balance: if provided, use proportional thresholds
    """
    if not records:
        return 0, 0, 0, 0, 0, 0
    
    # Basic metrics
    total_pnl = sum(r.get('pnl', 0) for r in records)
    num_trades = len(records)
    
    # Calculate running drawdown
    run_sum = 0.0
    max_run_loss = 0.0
    drawdowns = []
    
    for t in records:
        pnl_i = t.get('pnl', 0.0)
        run_sum += pnl_i
        
        if run_sum > 0:
            run_sum = 0.0
        
        if run_sum < max_run_loss:
            max_run_loss = run_sum
        
        if run_sum < 0:
            drawdowns.append(-run_sum)
    
    # Maximum consecutive loss (MCL)
    mcl = abs(max_run_loss)
    
    # Maximum drawdown percentage
    # If initial_balance provided, calculate as percentage
    if initial_balance and initial_balance > 0:
        mdd = mcl / initial_balance
        mcl_ratio = mcl / initial_balance
    else:
        # Fallback to your original logic
        mdd = mcl / 10000.0  # assuming some default
        mcl_ratio = mcl / 10000.0
    
    # Calculate score with proportional penalties
    if initial_balance:
        # Penalize based on percentage of initial balance
        drawdown_penalty = 1.0
        if mdd > 0.20:  # 20% drawdown
            drawdown_penalty *= 0.8
        if mdd > 0.30:  # 30% drawdown
            drawdown_penalty *= 0.7
        if mdd > 0.50:  # 50% drawdown
            drawdown_penalty *= 0.5
            
        score = (total_pnl * drawdown_penalty) / (1 + mdd)
    else:
        # Original scoring logic
        score = total_pnl / (1 + mcl/10000.0)
    
    return score, total_pnl, num_trades, mdd, mcl, mcl_ratio

def tune_rf_gemini(
    X_train, y_train,
    labeled,  # The full DataFrame for the backtester
    market: str,
    n_trials: int = 50,
    unique_id: str = None,
    n_jobs: int = -1
) -> dict:
    """
    Optimize RandomForestClassifier using the custom utility score within Optuna.
    """
    study_name = f"rf_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"

    # It's good practice to handle timezone outside the objective function
    if not isinstance(labeled.index, pd.DatetimeIndex):
        if "datetime" in labeled.columns:
            labeled = labeled.set_index("datetime")
        else:
            raise ValueError("No 'datetime' column to set as index!")
    if labeled.index.tz is None:
        labeled = labeled.tz_localize("UTC")
    # Using America/New_York as in your original code
    labeled = labeled.tz_convert("America/New_York")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 5000, step=10),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 50, 1000, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 200, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.1, 0.2, 0.5, 0.8]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'bootstrap': True,
            'oob_score': False
        }

        tscv = auto_ts_split(len(y_train))
        fold_scores     = []
        fold_mdds       = []
        fold_mcls       = []
        fold_exceeds    = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(**params, random_state=42, n_jobs=n_jobs)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            # backtest
            labeled_fold = labeled.loc[X_val.index]
            back = evaluate_classification(
                X_val, preds, labeled_fold,
                avoid_funcs={}, TRAIL_START_MULT=0,
                TRAIL_STOP_MULT=0, TICK_VALUE=6
            )
            results = back['results']
            records = results.to_dict('records') if isinstance(results, pd.DataFrame) else results

            # compute basic metrics
            score, pnl, nt, mdd, mcl, mcl_ratio = calculate_custom_score(records)

            # now count within-fold loss streak events over starting capital
            # rebuild the run_sum series to capture each drawdown event
            run_sum = 0.0
            exceed_count = 0
            for t in records:
                pnl_i = t.get('pnl', 0.0)
                run_sum += pnl_i
                if run_sum > 0:
                    run_sum = 0.0
                if -run_sum > 2000.0:  # starting capital threshold
                    exceed_count += 1

            print(f"Fold score: {score:.2f}, PnL: {pnl:.2f}, Trades: {nt}, "
                  f"MDD: {mdd:.1%}, MCL: {mcl:.2f}, Exceeds: {exceed_count}")

            fold_scores.append(score)
            fold_mdds.append(mdd)
            fold_mcls.append(mcl)
            fold_exceeds.append(exceed_count)

        mean_score  = float(np.mean(fold_scores))
        # penalize by total exceed events across folds
        total_exceeds = sum(fold_exceeds)
        penalized_score = mean_score / (1 + total_exceeds)

        trial.set_user_attr('mean_score',     mean_score)
        trial.set_user_attr('mean_mdd',       float(np.mean(fold_mdds)))
        trial.set_user_attr('mean_mcl',       float(np.mean(fold_mcls)))
        trial.set_user_attr('total_exceeds',  int(total_exceeds))

        return penalized_score

    study = create_study(study_name=study_name, direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    print(f"Best penalized score: {best.value:.2f}")
    print("Params:", best.params)
    print("Mean MDD:", best.user_attrs['mean_mdd'])
    print("Mean MCL:", best.user_attrs['mean_mcl'])
    print("Total exceed count:", best.user_attrs['total_exceeds'])
    return best.params

def tune_rf_gemini_crypto(
    X_train, y_train,
    labeled,
    market: str,
    n_trials: int = 50,
    unique_id: str = None,
    n_jobs: int = -1,
    leverage: float = 3.0,
    risk_fraction: float = 0.02,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
    initial_balance: float = 100.0,
    exceed_threshold: float = 0.40
):
    """
    Optimize RandomForestClassifier using the custom utility score within Optuna.
    """
    study_name = f"rf_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"

    # It's good practice to handle timezone outside the objective function
    if not isinstance(labeled.index, pd.DatetimeIndex):
        if "datetime" in labeled.columns:
            labeled = labeled.set_index("datetime")
        else:
            raise ValueError("No 'datetime' column to set as index!")
    if labeled.index.tz is None:
        labeled = labeled.tz_localize("UTC")
    # Using America/New_York as in your original code
    labeled = labeled.tz_convert("America/New_York")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 5000, step=10),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 50, 1000, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 200, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.1, 0.2, 0.5, 0.8]),
            'class_weight': 'balanced',
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'bootstrap': True,
            'oob_score': False
        }

        tscv = auto_ts_split(len(y_train))
        fold_scores     = []
        fold_mdds       = []
        fold_mcls       = []
        fold_exceeds    = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(**params, random_state=42, n_jobs=n_jobs)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            # Backtest with FIXED trading parameters
            labeled_fold = labeled.loc[X_val.index]
            
            back = evaluate_crypto_classification(
                X_val, preds, labeled_fold,
                avoid_funcs={}, 
                TRAIL_START_MULT=0,
                TRAIL_STOP_MULT=0,
                initial_balance=initial_balance,
                leverage=leverage,  # Fixed leverage
                RISK_FRACTION=risk_fraction,  # Fixed risk
                SL_ATR_MULT=sl_mult,  # Fixed SL
                TP_ATR_MULT=tp_mult,  # Fixed TP
            )
            
            results = back['results']
            records = results.to_dict('records') if isinstance(results, pd.DataFrame) else results

            # Calculate MDD and exceed count based on peak balance
            peak_balance = initial_balance
            current_balance = initial_balance
            max_drawdown = 0.0
            exceed_count = 0
            
            for t in records:
                pnl_i = t.get('pnl', 0.0)
                current_balance += pnl_i
                
                if current_balance > peak_balance:
                    peak_balance = current_balance
                
                if peak_balance > 0:
                    drawdown = (peak_balance - current_balance) / peak_balance
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    if drawdown > exceed_threshold:
                        exceed_count += 1

            # Your original scoring
            score, pnl, nt, _, mcl, mcl_ratio = calculate_custom_score(records)

            print(f"Fold score: {score:.2f}, PnL: {pnl:.2f}, Trades: {nt}, "
                  f"MDD: {max_drawdown:.1%}, MCL: {mcl:.2f}, Exceeds: {exceed_count}")

            fold_scores.append(score)
            fold_mdds.append(max_drawdown)
            fold_mcls.append(mcl)
            fold_exceeds.append(exceed_count)

        mean_score = float(np.mean(fold_scores))
        total_exceeds = sum(fold_exceeds)
        penalized_score = mean_score / (1 + total_exceeds)

        trial.set_user_attr('mean_score', mean_score)
        trial.set_user_attr('mean_mdd', float(np.mean(fold_mdds)))
        trial.set_user_attr('mean_mcl', float(np.mean(fold_mcls)))
        trial.set_user_attr('total_exceeds', int(total_exceeds))

        return penalized_score

    study = create_study(study_name=study_name, direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    print(f"Best penalized score: {best.value:.2f}")
    print("Params:", best.params)
    print("Mean MDD:", best.user_attrs['mean_mdd'])
    print("Mean MCL:", best.user_attrs['mean_mcl'])
    print("Total exceed count:", best.user_attrs['total_exceeds'])
    
    return best.params

def tune_catboost(
    X_train, y_train,
    market: str,
    n_trials: int = 50,
    unique_id: str = None
) -> dict:
    """
    Optimize CatBoostClassifier using weighted CV and training-set F1 (80/20).
    """
    study_name = f"catboost_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"

    def objective(trial):
        bootstrap = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1500, step=100),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.5, 5.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'bootstrap_type': bootstrap,
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1',
            'class_weights': weights.tolist(),
            'verbose': 0,
            'random_state': 42
        }
        if bootstrap == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 1.0)
        tscv = auto_ts_split(len(y_train))
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=compute_sample_weight(class_weight='balanced', y=y_tr))
            preds = model.predict(X_val)
            cv_scores.append(f1_score(y_val, preds, average='macro'))
        cv_mean = np.mean(cv_scores)
        final = CatBoostClassifier(**params)
        final.fit(X_train, y_train, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train))
        train_f1 = f1_score(y_train, final.predict(X_train), average='macro')
        score = 0.8 * cv_mean + 0.2 * train_f1
        trial.set_user_attr('cv_mean', cv_mean)
        trial.set_user_attr('train_f1', train_f1)
        return score

    study = create_study(
        study_name=study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_logistic_regression(
    X_train, y_train,
    market: str,
    n_trials: int = 20,
    unique_id: str = None
) -> dict:
    """
    Optimize LogisticRegression (L2) using cross-val + training F1 blend.
    """
    study_name = f"logreg_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"

    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        params = {
            'C': C,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'multi_class': 'auto'
        }
        tscv = auto_ts_split(len(y_train))
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            weights = compute_sample_weight(class_weight='balanced', y=y_tr)
            model = LogisticRegression(**params)
            model.fit(X_tr, y_tr, sample_weight=weights)
            preds = model.predict(X_val)
            cv_scores.append(f1_score(y_val, preds, average='macro'))
        cv_mean = np.mean(cv_scores)
        final = LogisticRegression(**params)
        final.fit(X_train, y_train, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train))
        train_f1 = f1_score(y_train, final.predict(X_train), average='macro')
        score = 0.8 * cv_mean + 0.2 * train_f1
        trial.set_user_attr('cv_mean', cv_mean)
        trial.set_user_attr('train_f1', train_f1)
        return score

    study = create_study(
        study_name=study_name,
        direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
