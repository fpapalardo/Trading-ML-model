import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_sample_weight, compute_class_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

def tune_xgboost(
    X_train, y_train,
    market: str,
    lookahead: int,
    splits: int = 4,
    n_trials: int = 50,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize an XGBClassifier using weighted CV and training-set F1 (80/20).
    """
    study_name = f"xgb_opt_class_{market}_{lookahead}"
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
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train)),
            'eval_metric': 'logloss'
        }
        tscv = TimeSeriesSplit(n_splits=splits)
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            weights = compute_sample_weight(class_weight='balanced', y=y_tr)
            model = XGBClassifier(**params, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr, sample_weight=weights)
            preds = model.predict(X_val)
            cv_scores.append(f1_score(y_val, preds, average='macro'))
        cv_mean = np.mean(cv_scores)
        final = XGBClassifier(**params, random_state=42, n_jobs=-1)
        final.fit(X_train, y_train)
        train_f1 = f1_score(y_train, final.predict(X_train), average='macro')
        score = 0.8 * cv_mean + 0.2 * train_f1
        trial.set_user_attr('cv_mean', cv_mean)
        trial.set_user_attr('train_f1', train_f1)
        return score

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_lgbm(
    X_train, y_train,
    market: str,
    splits: int = 4,
    n_trials: int = 50,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize an LGBMClassifier using weighted CV and training-set F1 (80/20).
    """
    study_name = f"lgbm_opt_class_{market}"
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
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        tscv = TimeSeriesSplit(n_splits=splits)
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

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_rf(
    X_train, y_train,
    market: str,
    splits: int = 4,
    n_trials: int = 50,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize RandomForestClassifier using cross-val + OOB blend.
    """
    study_name = f"rf_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"
    storage = f"sqlite:///{db_dir}/{study_name}.db"

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 100, 300, step=50),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 100, step=10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50, step=5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.2, 0.5, 0.8]),
            'bootstrap': True,
            'oob_score': True,
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
        tscv = TimeSeriesSplit(n_splits=splits)
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=tscv, scoring='f1_macro', n_jobs=-1
        )
        model.fit(X_train, y_train)
        oob = model.oob_score_
        score = 0.8 * cv_scores.mean() + 0.2 * oob
        trial.set_user_attr('cv_mean', cv_scores.mean())
        trial.set_user_attr('oob_score', oob)
        return score

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_catboost(
    X_train, y_train,
    market: str,
    splits: int = 4,
    n_trials: int = 50,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize CatBoostClassifier using weighted CV and training-set F1 (80/20).
    """
    study_name = f"catboost_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"
    storage = f"sqlite:///{db_dir}/{study_name}.db"

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
        tscv = TimeSeriesSplit(n_splits=splits)
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

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_logistic_regression(
    X_train, y_train,
    market: str,
    splits: int = 4,
    n_trials: int = 20,
    db_dir: str = "notebooks/dbs",
    unique_id: str = None
) -> dict:
    """
    Optimize LogisticRegression (L2) using cross-val + training F1 blend.
    """
    study_name = f"logreg_opt_class_{market}"
    if unique_id:
        study_name += f"_{unique_id}"
    storage = f"sqlite:///{db_dir}/{study_name}.db"

    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        params = {
            'C': C,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'multi_class': 'auto'
        }
        tscv = TimeSeriesSplit(n_splits=splits)
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

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
