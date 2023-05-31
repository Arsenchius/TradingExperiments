#!/bin/bash

DATA=/home/kenny/data
LOGGER=/home/kenny/TradingExperiments/src/logging.conf

EXECUTABLE=$1
EXP=$2

python $EXECUTABLE --data-dir-path $DATA --exp-id $EXP --logger-path $LOGGER
