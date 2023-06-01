import os
import math
import json
import warnings
import argparse
import datetime
from multiprocessing import Process
from typing import NoReturn, Tuple, Dict

from tqdm import tqdm
import pandas as pd
import numpy as np
import lightgbm as lgb

from clean import read_data, feature_creation

warnings.filterwarnings("ignore")

def _helper(df: pd.DataFrame, vol: float, fee: float) -> Tuple[int,int]:

    '''
    df - dataframe
    vol - volume for one order
    fee - comission for one order
    '''

    df["MidPrice"] = df[["px_buy_1", "px_sell_1"]].mean(axis=1)
    df["MidPricePrev"] = df["MidPrice"].shift(1).fillna(0)
    buy_chances = len(df[(df["MidPrice"] - df["MidPricePrev"]) > (2 * vol * df["MidPricePrev"] * fee / 100)])
    sell_chances = len(df[(df["MidPricePrev"] - df["MidPrice"]) > (2 * vol * df["MidPricePrev"] * fee / 100)])
    return buy_chances, sell_chances

'''
Time complexity for this execution is O(nlogn)
For one dataframe with size <= 3 * 10^6 elements,  execution time will be ~ 8 min
'''
def _helper_with_time_delay(df: pd.DataFrame, vol:float, fee: float, time_delay:int) -> Tuple[int,int]:

    '''
    df - dataframe
    vol - volume for one order
    fee - comission for one order
    time_delay - time delay for order execution in milliseconds
    '''

    df = read_data(df)
    df = feature_creation(df)
    long_chances, short_chances = 0,0
    for index, row in df.iterrows():
        timestamp = index
        delta = datetime.timedelta(milliseconds=time_delay)  # Create a timedelta representing 40 milliseconds
        new_timestamp = timestamp + delta  # Add the timedelta to the timestamp
        first_position = df.index.searchsorted(new_timestamp) # Time complexity is log(n) for searching
        if first_position >= len(df):
            continue
        mid_price_entry = row['TruePrice']
        mid_price_out = df.iloc[first_position]['TruePrice']
        if (mid_price_out - mid_price_entry) >= (2*vol*mid_price_entry*fee / 100):
            long_chances += 1
        elif (mid_price_entry - mid_price_out) >= (2*vol*mid_price_entry*fee / 100):
            short_chances += 1
    return long_chances, short_chances

def make_some_analysis(path:str,vol:float, time_delay:int,fee:float=0.1) -> Dict:

    '''
    path - path to a data file
    vol - volume of an order
    time_delay - time delay for order execution in milliseconds
    fee - default value for binance is 0.1%
    '''

    chunk_size = 1000000
    csv_reader = pd.read_csv(path, chunksize=chunk_size, sep="|", iterator=True)
    total_chunks = math.ceil(sum(1 for line in open(path)) / chunk_size)
    total_number_of_events = 0
    result = {
        "vol": vol,
        "total": 0,
        "buy_chances": 0,
        "sell_chances": 0,
        "time_delay": time_delay,
    }
    for part_number, chunk in enumerate(csv_reader):
        if part_number == 0:
            chunk = chunk.iloc[10:]
        elif part_number == total_chunks - 1:
            chunk = chunk.iloc[:-10]
        if time_delay != 0:
            buy_chances, sell_chances = _helper_with_time_delay(chunk, vol, fee, time_delay)
        else:
            buy_chances, sell_chances = _helper(chunk, vol, fee)
        total_number_of_events += len(chunk)
        result["buy_chances"] += buy_chances
        result["sell_chances"] += sell_chances
    result["total"] = total_number_of_events
    return result

def _run_part(date:str, snapshot_data_path:str, output_dir_path:str, time_delta:int) -> NoReturn:
    result = []
    for vol in tqdm(np.arange(0.01, 0.11, 0.01)):
        result.append(make_some_analysis(snapshot_data_path, vol=vol, time_delay=time_delta))
    full_output_path = os.path.join(output_dir_path, date+'.json')
    with open(full_output_path, 'w') as file:
        json.dump(result, file)

def _merge_results(output_dir_path:str) -> NoReturn:
    final_result = {}
    for part_path in os.listdir(output_dir_path):
        full_part_path = os.path.join(output_dir_path, part_path)
        with open(full_part_path, 'r') as file:
            part_result = json.load(file)
        date = part_path.split('.')[0]
        final_result[date] = part_result
    full_output_path = os.path.join(output_dir_path, 'overall.json')
    with open(full_output_path, 'w') as file:
        json.dump(final_result, file)

def run(args):
    data_dir = args.data_dir_path
    output_dir = args.output_dir_path
    pair_name = args.pair_name
    time_delta = args.time_delta
    data_dir_path = os.path.join(data_dir, pair_name.lower())
    if time_delta != 0:
        output_dir_path = os.path.join(output_dir, pair_name.lower()+"_time_delta_" + str(time_delta))
    else:
        output_dir_path = os.path.join(output_dir, pair_name.lower())
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    part_jobs = []
    for snapshot_data in os.listdir(data_dir_path):
        date = '_'.join(snapshot_data.split('/')[-1][:-4].split('_')[:0:-1])
        snapshot_data_path = os.path.join(data_dir_path, snapshot_data)
        part_jobs.append(Process(target=_run_part, args=(
            date,
            snapshot_data_path,
            output_dir_path,
            time_delta,
        )))

    for job in part_jobs:
        job.start()

    for job in part_jobs:
        job.join()

    _merge_results(output_dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir-path", type=str, help="Path to input data dir", required=True
    )
    parser.add_argument("--output-dir-path", type=str, help="Path to output dir", required=True)
    parser.add_argument("--pair-name", type=str, help="Pair name", required=True)
    parser.add_argument("--time-delta", type=int, help="Define a time delta for analysis", required=True)
    args = parser.parse_args()

    run(args)
