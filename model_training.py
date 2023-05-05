import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from objective import objective
import yaml
import json
import math
import random
from clean import read_data, feature_creation
from typing import NoReturn

import warnings
warnings.filterwarnings("ignore")



def tuning(path_to_data: str, path_to_params_config: str) -> NoReturn:
    chunk_size = 10 ** 6
    total_chunks = math.ceil(sum(1 for line in open(path_to_data)) / chunk_size)
    csv_reader = pd.read_csv(path_to_data, chunksize=chunk_size, sep="|", iterator=True)
    random_number_of_chunk = random.randint(1, total_chunks-1)
    for i, chunk in enumerate(csv_reader):
        if i == random_number_of_chunk:
            chunk = read_data(chunk)
            df = feature_creation(chunk)

    # df = data.drop(["host_time", "sent_time"], axis=1)
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


def _train(model, csv_reader, total_chunks):
    # Initialize variables to store overall MSE and total number of samples
    overall_mse, total_samples = .0, 0

    for i, chunk in enumerate(csv_reader):
        if i == 0:
            chunk = chunk.iloc[10:]
        elif i == total_chunks - 1:
            chunk = chunk.iloc[:-10]
        chunk = read_data(chunk)
        chunk = feature_creation(chunk)

        X_chunk = chunk.drop(columns=['TruePrice'])
        y_chunk = chunk['TruePrice']

        model.fit(X_chunk, y_chunk, eval_set=[(X_chunk, y_chunk)], eval_metric='mse', verbose=False)

        # Predict target variable for chunk of data
        preds_chunk = model.predict(X_chunk)
        # Calculate MSE for chunk of data
        mse_chunk = mean_squared_error(y_chunk, preds_chunk)

        # Update overall MSE and total number of samples
        overall_mse += mse_chunk * len(chunk)
        total_samples += len(chunk)

    return model, overall_mse, total_samples


def model_training(
    path_to_current_day_data: str,
    path_to_previous_day_data: str,
    path_to_params_config: str,
    path_to_model_json: str,
    path_to_model_txt: str
    )-> NoReturn:
    chunk_size = 1000000
    csv_reader_current_day = pd.read_csv(path_to_current_day_data, chunksize=chunk_size, sep="|", iterator=True)
    total_chunks_current_day = math.ceil(sum(1 for line in open(path_to_current_day_data)) / chunk_size)

    csv_reader_previous_day = pd.read_csv(path_to_previous_day_data, chunksize=chunk_size, sep="|", iterator=True)
    total_chunks_previous_day = math.ceil(sum(1 for line in open(path_to_previous_day_data)) / chunk_size)

    with open(path_to_params_config, 'r') as file:
        best_params = yaml.safe_load(file)

    lgb_model = lgb.LGBMRegressor(**best_params)

    lgb_model, overall_mse_current, total_samples_current = _train(lgb_model, csv_reader_current_day, total_chunks_current_day)
    lgb_model, overall_mse_previous, total_samples_previous = _train(lgb_model, csv_reader_previous_day, total_chunks_previous_day)

    # Calculate weighted average of MSEs for all chunks
    overall_mse = (overall_mse_current + overall_mse_previous) / (total_samples_current + total_chunks_previous_day)

    print(f'Overall MSE: {overall_mse}')

    # Save trained model to JSON file
    model_json = json.dumps(lgb_model.booster_.dump_model())
    with open(path_to_model_json, 'w') as f:
        f.write(model_json)

    # Save trained model to txt file
    lgb_model.booster_.save_model(path_to_model_txt)
