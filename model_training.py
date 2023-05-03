import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from objective import objective
import yaml

def tuning(data, path_to_params_config):
    df = data.drop(["host_time", "sent_time"], axis=1)
    features = list(df.columns)
    features.remove('TruePrice')

    # Run hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective(trial, df, features)
    study.optimize(func, n_trials=1)

    # Train final model using best hyperparameters
    best_params = study.best_params
    X_train = df[features]
    y_train = df['TruePrice']

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(best_params, train_data, valid_sets=lgb.Dataset(X_train, y_train), verbose_eval=False)

    # Evaluate model
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    print('Best model trained with MSE:', mse)

    # Save best params to YAML file
    with open(path_to_params_config, 'w') as f:
        yaml.dump(best_params, f)

    print('Best parameters saved to config file!')

