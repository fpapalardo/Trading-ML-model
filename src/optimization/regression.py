import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


def tune_xgboost_regressor(
    X_train, y_train,
    market: str,
    lookahead: int,
    splits: int = 4,
    n_trials: int = 50,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize XGBRegressor via TS-split CV and training-set RMSE blend.
    """
    study_name = f"xgb_opt_reg_{market}_{lookahead}"
    if unique_id:
        study_name += f"_{unique_id}"
    storage = f"sqlite:///{db_dir}/{study_name}.db"

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'objective': 'reg:squarederror'
        }
        tscv = TimeSeriesSplit(n_splits=splits)
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = XGBRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            cv_scores.append(rmse)
        cv_mean = np.mean(cv_scores)
        final = XGBRegressor(**params, random_state=42, n_jobs=-1)
        final.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, final.predict(X_train)))
        score = 0.8 * cv_mean + 0.2 * train_rmse
        trial.set_user_attr('cv_rmse', cv_mean)
        trial.set_user_attr('train_rmse', train_rmse)
        return score

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_lgbm_regressor(
    X_train, y_train,
    market: str,
    splits: int = 4,
    n_trials: int = 50,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize LGBMRegressor via TS-split CV and training-set RMSE blend.
    """
    study_name = f"lgbm_opt_reg_{market}"
    if unique_id:
        study_name += f"_{unique_id}"
    storage = f"sqlite:///{db_dir}/{study_name}.db"

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
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
        }
        tscv = TimeSeriesSplit(n_splits=splits)
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = LGBMRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        cv_mean = np.mean(cv_scores)
        final = LGBMRegressor(**params, random_state=42, n_jobs=-1)
        final.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, final.predict(X_train)))
        score = 0.8 * cv_mean + 0.2 * train_rmse
        trial.set_user_attr('cv_rmse', cv_mean)
        trial.set_user_attr('train_rmse', train_rmse)
        return score

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_rf_regressor(
    X_train, y_train,
    market: str,
    splits: int = 4,
    n_trials: int = 50,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize RandomForestRegressor via TS-split CV and training-set RMSE blend.
    """
    study_name = f"rf_opt_reg_{market}"
    if unique_id:
        study_name += f"_{unique_id}"
    storage = f"sqlite:///{db_dir}/{study_name}.db"

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        }
        tscv = TimeSeriesSplit(n_splits=splits)
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        cv_mean = np.mean(cv_scores)
        final = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        final.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, final.predict(X_train)))
        score = 0.8 * cv_mean + 0.2 * train_rmse
        trial.set_user_attr('cv_rmse', cv_mean)
        trial.set_user_attr('train_rmse', train_rmse)
        return score

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_catboost_regressor(
    X_train, y_train,
    market: str,
    splits: int = 4,
    n_trials: int = 50,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize CatBoostRegressor via TS-split CV and training-set RMSE blend.
    """
    study_name = f"catb_opt_reg_{market}"
    if unique_id:
        study_name += f"_{unique_id}"
    storage = f"sqlite:///{db_dir}/{study_name}.db"

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        tscv = TimeSeriesSplit(n_splits=splits)
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = CatBoostRegressor(**params, random_state=42, verbose=0)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        cv_mean = np.mean(cv_scores)
        final = CatBoostRegressor(**params, random_state=42, verbose=0)
        final.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, final.predict(X_train)))
        score = 0.8 * cv_mean + 0.2 * train_rmse
        trial.set_user_attr('cv_rmse', cv_mean)
        trial.set_user_attr('train_rmse', train_rmse)
        return score

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_ridge_regressor(
    X_train, y_train,
    market: str,
    splits: int = 4,
    n_trials: int = 20,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize Ridge regression via TS-split CV and training-set RMSE blend.
    """
    study_name = f"ridge_opt_reg_{market}"
    if unique_id:
        study_name += f"_{unique_id}"
    storage = f"sqlite:///{db_dir}/{study_name}.db"

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
        params = {'alpha': alpha, 'random_state': 42}
        tscv = TimeSeriesSplit(n_splits=splits)
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            model = Ridge(**params)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = model.predict(X_train.iloc[val_idx])
            cv_scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds)))
        cv_mean = np.mean(cv_scores)
        final = Ridge(**params)
        final.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, final.predict(X_train)))
        score = 0.8 * cv_mean + 0.2 * train_rmse
        trial.set_user_attr('cv_rmse', cv_mean)
        trial.set_user_attr('train_rmse', train_rmse)
        return score

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
