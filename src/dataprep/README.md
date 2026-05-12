# Data Preparation and Fusion

This module contains Bronze → Silver → Gold preprocessing pipelines for the MIMIC-III dataset, split across three contributor scripts.

---

## vhminh – Data Fusion (`vhminh_data_fusion.py`)

**Input tables:** `CHARTEVENTS`, `LABEVENTS`, `MICROBIOLOGYEVENTS`, `D_ITEMS`, `D_LABITEMS`

**Outputs:**

| Output | Description |
|---|---|
| `chart_events_named` | `CHARTEVENTS` joined with `D_ITEMS` |
| `lab_events_named` | `LABEVENTS` joined with `D_LABITEMS` |
| `microbiology_clean` | Cleaned `MICROBIOLOGYEVENTS` |

**Usage:**

```bash
bash src/dataprep/vhminh_fusion.sh
# or directly:
python src/dataprep/vhminh_data_fusion.py \
    --input-dir  ./data/raw \
    --output-dir ./data/processed \
    --spark-temp-dir ./.tmp/spark
```

---

## nxmanh – Procedures & CPT (`nxmanh_preprocessing.py`)

**Input tables:** `CPTEVENTS`, `PROCEDURES_ICD`, `PROCEDUREEVENTS_MV`, `D_CPT`, `D_ICD_PROCEDURES`

**Outputs (per table):**

| Layer | Files |
|---|---|
| Bronze | `bronze_cptevents`, `bronze_procedures_icd`, `bronze_procedureevents_mv`, `bronze_d_cpt`, `bronze_d_icd_procedures` |
| Silver | `silver_cptevents`, `silver_procedures_icd`, `silver_procedureevents_mv` |
| Gold | `gold_cptevents`, `gold_procedures_icd`, `gold_procedureevents_mv`, `gold_d_cpt`, `gold_d_icd_procedures` |

**Usage:**

```bash
bash src/dataprep/nxmanh_preprocessing.sh
# or directly:
python src/dataprep/nxmanh_preprocessing.py \
    --input-dir  ./data/raw \
    --output-dir ./data/processed \
    --spark-temp-dir ./.tmp/spark_nxmanh \
    --tables cptevents procedures_icd   # optional: run specific tables only
```

---

## trangptt – Output Events & Diagnoses (`trangptt_preprocessing.py`)

**Input tables:** `OUTPUTEVENTS`, `DIAGNOSES_ICD`, `D_ICD_DIAGNOSES`

**Outputs (per table):**

| Layer | Files |
|---|---|
| Bronze | `bronze_outputevents`, `bronze_diagnoses_icd`, `bronze_d_icd_diagnoses` |
| Silver | `silver_outputevents`, `silver_diagnoses_icd`, `silver_d_icd_diagnoses` |
| Gold | `gold_outputevents`, `gold_diagnoses_icd` |

> `gold_diagnoses_icd` là kết quả left-join `DIAGNOSES_ICD ← D_ICD_DIAGNOSES` trên `ICD9_CODE`.

**Usage:**

```bash
bash src/dataprep/trangptt_preprocessing.sh
# or directly:
python src/dataprep/trangptt_preprocessing.py \
    --input-dir  ./data/raw \
    --output-dir ./data/processed \
    --spark-temp-dir ./.tmp/spark_trangptt \
    --tables outputevents diagnoses_icd   # optional: run specific tables only
```
