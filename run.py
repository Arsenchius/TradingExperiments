import os
import time

# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
import argparse
import math
import random
from clean import read_data, feature_creation
from multiprocessing import Process
from datetime import datetime, timedelta
from model_training import tuning, model_training
from experiments import EXPERIMENT_ID_TO_PARAMETERS
from bt_run import backtest_run

# import warnings
# warnings.filterwarnings("ignore")


def run(args):
    print("Begin....")
    data_dir_path = args.data_dir_path
    exp_info = EXPERIMENT_ID_TO_PARAMETERS[args.exp_id]

    output_dir_path = exp_info["output_dir"]
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    model_best_params_path = os.path.join(output_dir_path, "model_best_params.yaml")
    model_path_json = os.path.join(output_dir_path, "model.json")
    model_path_txt = os.path.join(output_dir_path, "model.txt")

    current_day = exp_info["current_day"]
    pair_name = exp_info["pair_name"]

    # get a previous day datetime
    current_date_object = datetime.strptime(current_day, "%d.%m.%Y").date()
    previous_date_object = current_date_object - timedelta(days=1)
    previous_day = previous_date_object.strftime("%d.%m.%Y")

    current_day_splitted = current_day.split(".")
    previous_day_splitted = previous_day.split(".")

    # get a paths to current and previous days for model training
    current_day_path = os.path.join(
        data_dir_path,
        pair_name
        + "_"
        + current_day_splitted[-1]
        + "_"
        + current_day_splitted[-2]
        + "_"
        + current_day_splitted[0]
        + ".csv",
    )
    previous_day_path = os.path.join(
        data_dir_path,
        pair_name
        + "_"
        + previous_day_splitted[-1]
        + "_"
        + previous_day_splitted[-2]
        + "_"
        + previous_day_splitted[0]
        + ".csv",
    )

    print("All path sets correctly!")

    print("Start tuning...")
    tuning(current_day_path, model_best_params_path)
    print("Tuning model finished!")

    print("Start model training...")
    model_training(
        current_day_path, previous_day_path, model_best_params_path, model_path_json, model_path_txt
    )
    print("Model trained!")

    # start BackTest on other dates:
    # print("Run BackTest...")
    # backtest_run(model_path, current_day_path, previous_day_path, data_dir_path)

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
    args = parser.parse_args()

    run(args)
