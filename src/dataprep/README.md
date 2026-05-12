# Data Preparation and Fusion

This module is responsible for data preparation and fusion of the MIMIC-III dataset for five subsets of the corpus:
1. `CHARTEVENTS`
2. `LABEVENTS`
3. `MICROBIOLOGYEVENTS`
4. `D_ITEMS`
5. `D_LABITEMS`

## Overview
- The fusion/ merging events are described as follows:
    + `CHARTEVENTS + D_ITEMS` --> `chart_events_named`
    + `LABEVENTS + D_LABITEMS` --> `lab_events_named`
    + `MICROBIOLOGYEVENTS` --> `microbiology_clean`

## Usage

```bash
python src/data_fusion.py \
    --input-dir ./project_dir/data/raw \
    --output-dir ./project_dir/data/processed \
    --spark-temp-dir ./project_dir/.tmp/spark
```