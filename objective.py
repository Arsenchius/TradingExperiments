from sklearn.model_selection import TimeSeriesSplit
from typing import Callable, List
from optuna.trial import Trial
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd

# Define objective function for Optuna
def objective(
    trial: Trial,
    data: pd.DataFrame,
    features:List[str],
    number_of_splits: int = 5,
    test_size: float = 0.15
    ) -> float:
    # Define hyperparameters to optimize
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'num_boost_round': trial.suggest_int('num_boost_round', 10, 1000),
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 5, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'verbose': -1,
    }

    params["num_leaves"] = trial.suggest_categorical('num_leaves', [2 ** params['max_depth']]),

    # Train and evaluate model using time series cross-validation
    tss = TimeSeriesSplit(n_splits=number_of_splits, test_size=int(data.shape[0] * test_size))
    data = data.sort_index()
    rmse_scores = []
    for train_index, val_index in tss.split(data):
        train = data.iloc[train_index]
        test = data.iloc[val_index]

        X_train = train[features]
        y_train = train['LagTruePrice']

        X_test = test[features]
        y_test = test['LagTruePrice']

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test)

        model = lgb.train(params, train_data, valid_sets=[train_data, val_data], early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=False)
        y_pred = model.predict(X_test)
        rmse_scores.append(mean_squared_error(y_test, y_pred, squared=False))

    return sum(rmse_scores) / len(rmse_scores)

