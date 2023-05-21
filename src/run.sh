#!/bin/bash

DATA=/home/kenny/data

EXECUTABLE=$1
EXP=$2

python $EXECUTABLE --data-dir-path $DATA --exp-id $EXP
