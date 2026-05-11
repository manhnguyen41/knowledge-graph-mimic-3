#!/bin/bash

python src/data_fusion.py \
    --input-dir ./project_dir/data/raw \
    --output-dir ./project_dir/data/processed \
    --spark-temp-dir ./project_dir/.tmp/spark