#!/bin/bash

mkdir -p configs
mkdir -p configs/run
mkdir -p configs/finish

idx=(100 200 300 500 600 800 1000 1500 2000 2500 5000)

for t in ${idx[@]}; do
    echo "$t"
    
    sed -i -e 's/EVALUATION_CLASSIFICATION_BUDGET.*=.*/EVALUATION_CLASSIFICATION_BUDGET='"$t"'/g' config.py
    cp config.py configs/run/config"$t".py
done