#!/bin/bash

source activate RL

for entry in configs/run/*
do
    echo "$entry"
    cp $entry config.py
    python train.py
    mv $entry configs/finish/
done

