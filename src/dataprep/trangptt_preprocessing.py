"""
trangptt_preprocessing.py – Bronze → Silver → Gold Pipeline
Tables: OUTPUTEVENTS, DIAGNOSES_ICD, D_ICD_DIAGNOSES

Usage:
    python src/dataprep/trangptt_preprocessing.py \\
        --input-dir  /path/to/raw \\
        --output-dir /path/to/processed \\
        --spark-temp-dir /tmp/spark

    # Optional flags:
        --tables outputevents diagnoses_icd          # run specific tables only
        --master local[4]                            # Spark master
        --sample-limit 10000                         # quick test with N rows
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# WINDOWS: Set HADOOP_HOME trước khi import PySpark
# winutils.exe required: https://github.com/cdarlint/winutils
# ---------------------------------------------------------------------------
_HADOOP_HOME = r"C:\hadoop"
if sys.platform == "win32":
    os.environ.setdefault("HADOOP_HOME", _HADOOP_HOME)
    os.environ.setdefault("hadoop.home.dir", _HADOOP_HOME)
    _bin = os.path.join(_HADOOP_HOME, "bin")
    if _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType

# ---------------------------------------------------------------------------
# ALL SUPPORTED TABLES
# ---------------------------------------------------------------------------
ALL_TABLES = [
    "outputevents",
    "diagnoses_icd",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bronze→Silver→Gold preprocessing for trangptt's MIMIC-III tables."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing raw *.csv files.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where Parquet outputs will be saved.",
    )
    parser.add_argument(
        "--spark-temp-dir", default="/tmp/spark_trangptt",
        help="Spark local temp directory. Default: /tmp/spark_trangptt",
    )
    parser.add_argument(
        "--master", default="local[*]",
        help="Spark master URL. Default: local[*]",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=0,
        help="Row limit per table for quick testing (0 = full data).",
    )
    parser.add_argument(
        "--tables", nargs="+", choices=ALL_TABLES, default=ALL_TABLES,
        metavar="TABLE",
        help=(
            f"Tables to process. Choices: {ALL_TABLES}. "
            "Default: all tables."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SPARK SETUP
# ---------------------------------------------------------------------------

def build_spark(master: str, spark_temp_dir: str) -> SparkSession:
    os.makedirs(spark_temp_dir, exist_ok=True)
    warehouse_dir = os.path.join(os.path.dirname(spark_temp_dir), "warehouse")
    os.makedirs(warehouse_dir, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("trangptt_preprocessing")
        .master(master)
        .config("spark.local.dir", spark_temp_dir)
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def _path(directory: str, filename: str) -> str:
    return os.path.join(directory, filename)


def read_csv(spark: SparkSession, path: str, sample_limit: int = 0) -> DataFrame:
    """Đọc CSV với inferSchema. Áp sample_limit nếu > 0."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = spark.read.csv(path, header=True, inferSchema=True)
    if sample_limit > 0:
        df = df.limit(sample_limit)
    return df


def write_parquet(df: DataFrame, output_dir: str, name: str, partitions: int = 50) -> None:
    """Ghi DataFrame thành Parquet với số partition chỉ định."""
    out = _path(output_dir, name)
    print(f"[WRITE] {name} → {out}")
    df.repartition(partitions).write.mode("overwrite").parquet(out)


def report_null_counts(df: DataFrame, table_name: str) -> None:
    """In báo cáo Null cho tất cả cột (1 Spark Job). Highlight cột có null."""
    total = df.count()
    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    null_row = df.select(*null_exprs).collect()[0]
    sep = "-" * 65
    print(f"\n[NULL CHECK – {table_name}] {total:,} dòng")
    print(sep)
    print(f"{'TÊN CỘT':<25} | {'SỐ DÒNG NULL':<15} | TỶ LỆ %")
    print(sep)
    for col_name in df.columns:
        n = null_row[col_name]
        pct = n / total * 100 if total > 0 else 0.0
        ann = "  <-- Chú ý" if n > 0 else ""
        print(f"{col_name:<25} | {n:<15} | {pct:.2f}%{ann}")
    print(sep)


# ---------------------------------------------------------------------------
# TABLE: OUTPUTEVENTS  (Bronze → Silver → Gold)
# ---------------------------------------------------------------------------

def run_outputevents(spark: SparkSession, input_dir: str, output_dir: str,
                     sample_limit: int) -> None:
    print("\n" + "=" * 65)
    print("  TABLE: OUTPUTEVENTS")
    print("=" * 65)

    # Bronze
    df_bronze = read_csv(spark, _path(input_dir, "OUTPUTEVENTS.csv"), sample_limit)
    write_parquet(df_bronze, output_dir, "bronze_outputevents", partitions=50)
    report_null_counts(df_bronze, "OUTPUTEVENTS")

    # Silver: dropna key cols, normalize VALUEUOM, report duplicates
    dup_count = df_bronze.count() - df_bronze.dropDuplicates().count()
    print(f"[SILVER] Duplicate rows: {dup_count:,}")
    df_silver = (
        df_bronze
        .dropna(subset=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME"])
        .withColumn(
            "VALUEUOM",
            F.when(F.col("VALUEUOM") == "mL", "ml").otherwise(F.col("VALUEUOM"))
        )
    )
    write_parquet(df_silver, output_dir, "silver_outputevents", partitions=50)
    print(f"[SILVER] ✓ {df_silver.count():,} dòng")

    # Gold: convert timestamps, cast nullable IDs
    df_gold = (
        df_silver
        .withColumn("CHARTTIME", F.to_timestamp("CHARTTIME", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("STORETIME",  F.to_timestamp("STORETIME",  "yyyy-MM-dd HH:mm:ss"))
        .withColumn("HADM_ID",    F.col("HADM_ID").cast(LongType()))
        .withColumn("ICUSTAY_ID", F.col("ICUSTAY_ID").cast(LongType()))
    )
    write_parquet(df_gold, output_dir, "gold_outputevents", partitions=50)
    print(f"[GOLD] ✓ {df_gold.count():,} dòng")


# ---------------------------------------------------------------------------
# TABLE: DIAGNOSES_ICD  (Bronze → Silver → Gold)
# ---------------------------------------------------------------------------

def _normalize_col(col):
    return F.regexp_replace(
        F.regexp_replace(
            F.trim(F.lower(col)),
            r'[^\w\s<>+]', ''
        ),
        r'\s+', ' '
    )


def run_diagnoses_icd(spark: SparkSession, input_dir: str, output_dir: str,
                      sample_limit: int) -> None:
    print("\n" + "=" * 65)
    print("  TABLE: DIAGNOSES_ICD + D_ICD_DIAGNOSES")
    print("=" * 65)

    # Bronze
    df_diag_bronze  = read_csv(spark, _path(input_dir, "DIAGNOSES_ICD.csv"),   sample_limit)
    df_d_icd_bronze = read_csv(spark, _path(input_dir, "D_ICD_DIAGNOSES.csv"), sample_limit)
    write_parquet(df_diag_bronze,  output_dir, "bronze_diagnoses_icd",   partitions=50)
    write_parquet(df_d_icd_bronze, output_dir, "bronze_d_icd_diagnoses", partitions=10)
    report_null_counts(df_diag_bronze,  "DIAGNOSES_ICD")
    report_null_counts(df_d_icd_bronze, "D_ICD_DIAGNOSES")

    # Silver: normalize text titles, fix duplicate ICD9_CODE 75539 → 75529
    df_d_icd_silver = (
        df_d_icd_bronze
        .withColumn("long_title_clean",  _normalize_col(F.col("LONG_TITLE")))
        .withColumn("short_title_clean", _normalize_col(F.col("SHORT_TITLE")))
        .filter(F.col("ICD9_CODE") != "75539")
    )
    df_diag_silver = df_diag_bronze.withColumn(
        "ICD9_CODE",
        F.when(F.col("ICD9_CODE") == "75539", "75529").otherwise(F.col("ICD9_CODE"))
    )
    write_parquet(df_diag_silver,  output_dir, "silver_diagnoses_icd",   partitions=50)
    write_parquet(df_d_icd_silver, output_dir, "silver_d_icd_diagnoses", partitions=10)
    print(f"[SILVER] ✓ diagnoses_icd:   {df_diag_silver.count():,} dòng")
    print(f"[SILVER] ✓ d_icd_diagnoses: {df_d_icd_silver.count():,} bản ghi")

    # Gold: left-join diagnoses_icd ← d_icd_diagnoses on ICD9_CODE
    df_gold = df_diag_silver.join(
        df_d_icd_silver.drop("ROW_ID"), on="ICD9_CODE", how="left"
    )
    write_parquet(df_gold, output_dir, "gold_diagnoses_icd", partitions=50)
    print(f"[GOLD] ✓ {df_gold.count():,} dòng sau join")


# ---------------------------------------------------------------------------
# DISPATCHER
# ---------------------------------------------------------------------------

_RUNNERS = {
    "outputevents":  run_outputevents,
    "diagnoses_icd": run_diagnoses_icd,
}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    spark = build_spark(args.master, args.spark_temp_dir)

    print("=" * 65)
    print("  trangptt – MIMIC-III Preprocessing Pipeline")
    print(f"  Input : {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Tables: {args.tables}")
    print("=" * 65)

    for table in args.tables:
        _RUNNERS[table](spark, args.input_dir, args.output_dir, args.sample_limit)

    print("\n" + "=" * 65)
    print("  PIPELINE HOÀN THÀNH!")
    print("=" * 65)

    spark.stop()


if __name__ == "__main__":
    main()
