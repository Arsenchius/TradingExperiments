import pandas as pd
import lightgbm as lgb
import os
import json
import math
from tqdm import tqdm
from clean import read_data, feature_creation
from strategy import Strategy
from backtest import Backtest
import optuna
from optuna.trial import Trial
from typing import NoReturn

def _process(
    path_to_data: str,
    path_to_model: str,
    params: dict,
    latency: float = 0.2,
    fee:float = 0.002
    ) -> dict:
    chunk_size = 1000000
    csv_reader = pd.read_csv(path_to_data, chunksize=chunk_size, sep="|", iterator=True)
    total_chunks = math.ceil(sum(1 for line in open(path_to_data)) / chunk_size)

    # Load the model using the model saved file
    model = lgb.Booster(model_file=path_to_model)

    aggr_strategy = Strategy(model, threshold = params['threshold'], size= params['size'], max_position=params['max_position'], adj_coeff=params["adj_coef"])
    bt = Backtest(aggr_strategy)

    for i, chunk in enumerate(csv_reader):
        if i == 0:
            chunk = chunk.iloc[10:]
        elif i == total_chunks - 1:
            chunk = chunk.iloc[:-10]
        chunk = read_data(chunk)
        chunk = feature_creation(chunk)
        bt.run_backtest(data=chunk, latency=latency, fee=fee)

    return bt.summary(fee=fee)

def objective_function(
    trial: Trial,
    model_path: str,
    path_to_data: str
    ) -> float:

    # Define the search space for the input parameter
    parameters = {
        "threshold": trial.suggest_int('threshold', 20000, 40000),
        "size": trial.suggest_uniform('size', 0.1, 1.0),
        "max_position": trial.suggest_uniform('max_position', 1.0, 1.5),
        "adj_coef": trial.suggest_int('adj_coef', 100, 1000),
    }

    # Evaluate the output of the function
    output = _process(path_to_data, model_path, parameters)

    return output["returns"]

def parameters_optimization(
    model_path: str,
    path_to_data: str,
    strategy_params_path: str
    ) -> NoReturn:
    # Load the model using the model saved file
    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective_function(trial, model_path, path_to_data)
    study.optimize(func, n_trials=1)

    print(f"Best value: {study.best_value}, Best parameters: {study.best_params}")

    with open(strategy_params_path, 'w') as f:
        json.dump(study.best_params, f)


def backtest_run(
    model_path: str,
    path_to_current_day_data: str,
    path_to_previous_day_data: str,
    data_dir_path: str,
    backtest_results_path: str,
    strategy_params_path: str
    ) -> NoReturn:
    all_data_paths = [os.path.join(data_dir_path, x) for x in os.listdir(data_dir_path)]
    with open(strategy_params_path, 'r') as f:
        strategy_params = json.load(f)
    results = {}
    for data_path in tqdm(all_data_paths):
        if data_path not in [path_to_current_day_data, path_to_previous_day_data]:
            result = _process(data_path, model_path, strategy_params)
            data = ".".join(data_path[-14:-4].split("_")[::-1])
            results[data] = result

    with open(backtest_results_path, 'w') as f:
        json.dump(results, f)

