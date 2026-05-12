#!/bin/bash

python src/dataprep/nxmanh_preprocessing.py \
    --input-dir  ./data/raw \
    --output-dir ./data/processed \
    --spark-temp-dir ./.tmp/spark_nxmanh
