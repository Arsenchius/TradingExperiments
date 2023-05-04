import pandas as pd
import lightgbm as lgb
import os
import json
import math
from tqdm import tqdm


def _process(path_to_data, path_to_model):
    chunk_size = 1000000
    csv_reader = pd.read_csv(path_to_data, chunksize=chunk_size, sep="|", iterator=True)
    total_chunks = math.ceil(sum(1 for line in open(path_to_data)) / chunk_size)

    # Load the JSON file containing the model
    with open(path_to_model, 'r') as f:
        model_json = json.load(f)

    # Load the model using the JSON string
    model = lgb.Booster(model_file=None, model_str=json.dumps(model_json))
    
    pass

def backtest_run(model_path, path_to_current_day_data, path_to_previous_day_data, path_to_all_data):
    all_data_paths = [os.path.join(data_dir_path, x) for x in os.listdir(data_dir_path)]
    for data_path in tqdm(all_data_paths):
        if data_path not in [path_to_current_day_data, path_to_previous_day_data]:
            _process(data_path, model_path)

    pass
