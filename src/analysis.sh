#!/bin/bash

DATA=/home/kenny/data
OUTPUT=/home/kenny/TradingExperiments/reports/analysis_results

EXECUTABLE=$1
PAIRNAME=$2

python $EXECUTABLE --data-dir-path $DATA --output-dir-path $OUTPUT --pair-name $PAIRNAME
