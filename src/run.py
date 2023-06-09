import os
import time
import argparse
import math
import random
import logging
import logging.config
from multiprocessing import Process
from datetime import datetime, timedelta


# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np

from data.clean import read_data, feature_creation
from models.model_training import tuning, model_training
from experiments.experiments import EXPERIMENT_ID_TO_PARAMETERS
from backtest.bt_run import backtest_run, parameters_optimization


def _aggregate(date_: str, data_dir_path: str, pair_name: str) -> str:
    date_splitted = date_.split(".")
    return os.path.join(data_dir_path, pair_name+"_"+ date_splitted[-1]+ "_"+ date_splitted[-2]+ "_"+ date_splitted[0]+ ".csv")


def run(args):
    exp_info = EXPERIMENT_ID_TO_PARAMETERS[args.exp_id]
    data_dir_path = os.path.join(args.data_dir_path, exp_info["pair_name"].lower())
    logger_file = args.logger_path

    output_dir_path = exp_info["output_dir"]
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    log_file_path = os.path.join(output_dir_path, "execution.log")
    logging.config.fileConfig(logger_file, {'log_file_path': log_file_path})
    logger = logging.getLogger()

    logging.info('Begin execution')
    logging.info("Start setting paths...")
    model_best_params_path = os.path.join(output_dir_path, "model_best_params.yaml")
    model_path_json = os.path.join(output_dir_path, "lgb_model.json")
    model_path_txt = os.path.join(output_dir_path, "lgb_model.txt")
    strategy_best_params_path = os.path.join(output_dir_path, "strategy_params.json")
    backtest_results_path = os.path.join(output_dir_path, "backtest_results.json")

    current_day = exp_info["current_day"]
    pair_name = exp_info["pair_name"]
    date_for_strategy_tuning = exp_info["date_for_strategy_params_tuning"]

    # get a previous day datetime
    current_date_object = datetime.strptime(current_day, "%d.%m.%Y").date()
    previous_date_object = current_date_object - timedelta(days=1)
    previous_day = previous_date_object.strftime("%d.%m.%Y")

    current_day_splitted = current_day.split(".")
    previous_day_splitted = previous_day.split(".")
    date_for_strategy_tuning_splitted = date_for_strategy_tuning.split(".")

    # get a paths to current and previous days for model training
    current_day_path = _aggregate(current_day, data_dir_path, pair_name)
    previous_day_path = _aggregate(previous_day, data_dir_path, pair_name)
    data_for_strategy_tuning_path = _aggregate(date_for_strategy_tuning, data_dir_path, pair_name)

    logging.info('All path sets correctly!')

    logging.info('Start tuning...')
    tuning(current_day_path, model_best_params_path, log_file_path, logger_file)
    logging.info('Tuning model finished!')

    logging.info('Start model training...')
    model_training(
        current_day_path, previous_day_path, data_for_strategy_tuning_path, model_best_params_path, output_dir_path, log_file_path, logger_file, model_name="LGBM"
    )
    logging.info('Model trained!')

    # logging.info('Start strategy parameters tuning')
    # parameters_optimization(model_path_txt, data_for_strategy_tuning_path, strategy_best_params_path, log_file_path)
    # logging.info('Strategy parameters tuning finished!')

    # # start BackTest on other dates:
    # logging.info('Run BackTest...')
    # backtest_run(model_path_txt, current_day_path, previous_day_path, data_dir_path, backtest_results_path, strategy_best_params_path, log_file_path)
    # logging.info('Backtest finished!')

    # part_jobs = []
    # for part_name in os.listdir(test_dir_path):
    #     full_part_test_path = os.path.join(test_dir_path, part_name)
    #     full_part_output_path = os.path.join(output_dir_path, f'res-{part_name}')

    #     part_jobs.append(Process(target=_run_part, args=(
    #         full_part_test_path,
    #         full_part_output_path,
    #         completions_path,
    #         exp_id,
    #         model_path
    #     )))

    # for job in part_jobs:
    #     job.start()

    # for job in part_jobs:
    #     job.join()

    # _merge_results(output_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir-path", type=str, help="Path to input data dir", required=True
    )
    parser.add_argument("--exp-id", type=int, help="Id of experiment", required=True)
    parser.add_argument("--logger-path", type=str, help="Path to logger config file", required=True)
    args = parser.parse_args()

    run(args)
