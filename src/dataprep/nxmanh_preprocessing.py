"""
nxmanh_preprocessing.py – Bronze → Silver → Gold Pipeline
Tables: CPTEVENTS, PROCEDURES_ICD, PROCEDUREEVENTS_MV, D_CPT, D_ICD_PROCEDURES

Usage:
    python src/dataprep/nxmanh_preprocessing.py \\
        --input-dir  /path/to/raw \\
        --output-dir /path/to/processed \\
        --spark-temp-dir /tmp/spark

    # Optional flags:
        --tables cptevents procedures_icd          # run specific tables only
        --master local[4]                          # Spark master
        --sample-limit 10000                       # quick test with N rows
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
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# ALL SUPPORTED TABLES
# ---------------------------------------------------------------------------
ALL_TABLES = [
    "cptevents",
    "procedures_icd",
    "procedureevents_mv",
    "d_cpt",
    "d_icd_procedures",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bronze→Silver→Gold preprocessing for nxmanh's MIMIC-III tables."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing raw *.csv.gz files.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where Parquet outputs will be saved.",
    )
    parser.add_argument(
        "--spark-temp-dir", default="/tmp/spark_nxmanh",
        help="Spark local temp directory. Default: /tmp/spark_nxmanh",
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
        .appName("nxmanh_preprocessing")
        .master(master)
        .config("spark.local.dir", spark_temp_dir)
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.sql.shuffle.partitions", "8")
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
    """Đọc CSV.gz với inferSchema. Áp sample_limit nếu > 0."""
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
# TABLE: CPTEVENTS  (Bronze → Silver → Gold)
# ---------------------------------------------------------------------------

def run_cptevents(spark: SparkSession, input_dir: str, output_dir: str,
                  sample_limit: int) -> None:
    print("\n" + "=" * 65)
    print("  TABLE: CPTEVENTS")
    print("=" * 65)

    # Bronze
    df_bronze = read_csv(spark, _path(input_dir, "CPTEVENTS.csv.gz"), sample_limit)
    write_parquet(df_bronze, output_dir, "bronze_cptevents", partitions=50)
    report_null_counts(df_bronze, "CPTEVENTS")

    # Silver: drop cột rác, dropna key
    df_silver = df_bronze.drop("CPT_SUFFIX", "DESCRIPTION") \
                         .dropna(subset=["SUBJECT_ID", "HADM_ID", "CPT_CD"])
    write_parquet(df_silver, output_dir, "silver_cptevents", partitions=50)
    print(f"[SILVER] ✓ {df_silver.count():,} dòng")

    # Duplicate report
    wspec = Window.partitionBy("HADM_ID", "CPT_CD").orderBy("ROW_ID")
    df_ranked = df_silver.withColumn("rank", F.row_number().over(wspec))
    dup = df_ranked.filter("rank > 1").count()
    total_s = df_silver.count()
    print(f"[DUP CHECK] Dòng trùng: {dup:,} / {total_s:,} ({dup/total_s*100:.2f}%)")

    # Gold: groupBy + count
    df_gold = (
        df_silver
        .groupBy("SUBJECT_ID", "HADM_ID", "CPT_CD", "SECTIONHEADER", "SUBSECTIONHEADER")
        .agg(F.count("*").alias("TICKET_COUNT"))
    )
    write_parquet(df_gold, output_dir, "gold_cptevents", partitions=50)
    print(f"[GOLD] ✓ {df_gold.count():,} edges")


# ---------------------------------------------------------------------------
# TABLE: PROCEDURES_ICD  (Bronze → Silver → Gold)
# ---------------------------------------------------------------------------

def run_procedures_icd(spark: SparkSession, input_dir: str, output_dir: str,
                       sample_limit: int) -> None:
    print("\n" + "=" * 65)
    print("  TABLE: PROCEDURES_ICD")
    print("=" * 65)

    # Bronze
    df_bronze = read_csv(spark, _path(input_dir, "PROCEDURES_ICD.csv.gz"), sample_limit)
    write_parquet(df_bronze, output_dir, "bronze_procedures_icd", partitions=50)
    report_null_counts(df_bronze, "PROCEDURES_ICD")

    # Silver
    df_silver = df_bronze.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
    write_parquet(df_silver, output_dir, "silver_procedures_icd", partitions=50)
    print(f"[SILVER] ✓ {df_silver.count():,} dòng")

    # Duplicate report (GroupBy approach)
    dup_cases = (
        df_silver.groupBy("HADM_ID", "ICD9_CODE").count()
        .filter("count > 1").count()
    )
    print(f"[DUP CHECK] Tổ hợp (HADM_ID, ICD9_CODE) bị lặp: {dup_cases:,}")

    # Gold: groupBy + count + min seq
    df_gold = (
        df_silver
        .groupBy("SUBJECT_ID", "HADM_ID", "ICD9_CODE")
        .agg(
            F.count("*").alias("PROC_COUNT"),
            F.min("SEQ_NUM").alias("PRIMARY_SEQ_NUM"),
        )
    )
    write_parquet(df_gold, output_dir, "gold_procedures_icd", partitions=50)
    print(f"[GOLD] ✓ {df_gold.count():,} edges")


# ---------------------------------------------------------------------------
# TABLE: PROCEDUREEVENTS_MV  (Bronze → Silver → Gold via dedup)
# ---------------------------------------------------------------------------

def run_procedureevents_mv(spark: SparkSession, input_dir: str, output_dir: str,
                           sample_limit: int) -> None:
    print("\n" + "=" * 65)
    print("  TABLE: PROCEDUREEVENTS_MV")
    print("=" * 65)

    # Bronze
    df_bronze = read_csv(spark, _path(input_dir, "PROCEDUREEVENTS_MV.csv.gz"), sample_limit)
    write_parquet(df_bronze, output_dir, "bronze_procedureevents_mv", partitions=50)
    report_null_counts(df_bronze, "PROCEDUREEVENTS_MV")

    # Silver: auto-drop cols > 50% null, dropna key cols
    total_rows = df_bronze.count()
    null_exprs = [
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df_bronze.columns
    ]
    null_counts = df_bronze.select(*null_exprs).collect()[0].asDict()
    cols_to_drop = [
        c for c, n in null_counts.items()
        if total_rows > 0 and n / total_rows > 0.5
    ]
    print(f"[SILVER] Tự động bỏ cột (>50% null): {cols_to_drop}")
    df_silver = df_bronze.drop(*cols_to_drop) \
                         .dropna(subset=["HADM_ID", "STARTTIME", "ITEMID"])
    write_parquet(df_silver, output_dir, "silver_procedureevents_mv", partitions=50)
    print(f"[SILVER] ✓ {df_silver.count():,} dòng – {len(df_silver.columns)} cột")

    # Gold: deduplication via Window ROW_NUMBER (FinishedRunning ưu tiên)
    df_with_priority = df_silver.withColumn(
        "priority",
        F.when(F.col("STATUSDESCRIPTION") == "FinishedRunning", 0).otherwise(1),
    )
    wspec = Window.partitionBy(
        "HADM_ID", "ITEMID", "VALUE", "CGID", "STARTTIME", "ENDTIME"
    ).orderBy("priority", "ROW_ID")
    df_gold = (
        df_with_priority
        .withColumn("rank", F.row_number().over(wspec))
        .filter("rank == 1")
        .drop("priority", "rank")
    )
    initial = df_silver.count()
    final = df_gold.count()
    write_parquet(df_gold, output_dir, "gold_procedureevents_mv", partitions=50)
    print(f"[GOLD] ✓ {final:,} dòng sau dedup (loại {initial - final:,} dòng trùng)")


# ---------------------------------------------------------------------------
# TABLE: D_CPT  (Bronze → Gold, dimension table)
# ---------------------------------------------------------------------------

def run_d_cpt(spark: SparkSession, input_dir: str, output_dir: str,
              sample_limit: int) -> None:
    print("\n" + "=" * 65)
    print("  TABLE: D_CPT (Dimension)")
    print("=" * 65)

    df_bronze = read_csv(spark, _path(input_dir, "D_CPT.csv.gz"), sample_limit)
    write_parquet(df_bronze, output_dir, "bronze_d_cpt", partitions=10)
    report_null_counts(df_bronze, "D_CPT")

    # Gold: dropna primary keys
    df_gold = df_bronze.dropna(subset=["CATEGORY", "MINCODEINSUBSECTION", "MAXCODEINSUBSECTION"])
    write_parquet(df_gold, output_dir, "gold_d_cpt", partitions=10)
    print(f"[GOLD] ✓ {df_gold.count():,} bản ghi từ điển CPT")

    print("\n[MẪU] Cấu trúc phân cấp D_CPT:")
    df_gold.select(
        "CATEGORY", "SECTIONHEADER", "SUBSECTIONHEADER",
        "MINCODEINSUBSECTION", "MAXCODEINSUBSECTION",
    ).orderBy("MINCODEINSUBSECTION").show(10, truncate=False)


# ---------------------------------------------------------------------------
# TABLE: D_ICD_PROCEDURES  (Bronze → Gold, dimension table)
# ---------------------------------------------------------------------------

def run_d_icd_procedures(spark: SparkSession, input_dir: str, output_dir: str,
                         sample_limit: int) -> None:
    print("\n" + "=" * 65)
    print("  TABLE: D_ICD_PROCEDURES (Dimension)")
    print("=" * 65)

    df_bronze = read_csv(spark, _path(input_dir, "D_ICD_PROCEDURES.csv.gz"), sample_limit)
    write_parquet(df_bronze, output_dir, "bronze_d_icd_procedures", partitions=10)
    report_null_counts(df_bronze, "D_ICD_PROCEDURES")

    df_gold = df_bronze.dropna(subset=["ICD9_CODE"])
    write_parquet(df_gold, output_dir, "gold_d_icd_procedures", partitions=10)
    print(f"[GOLD] ✓ {df_gold.count():,} bản ghi từ điển ICD-9")

    print("\n[MẪU] Từ điển thủ thuật ICD-9:")
    df_gold.select("ICD9_CODE", "SHORT_TITLE", "LONG_TITLE").show(10, truncate=False)


# ---------------------------------------------------------------------------
# DISPATCHER
# ---------------------------------------------------------------------------

_RUNNERS = {
    "cptevents":         run_cptevents,
    "procedures_icd":    run_procedures_icd,
    "procedureevents_mv": run_procedureevents_mv,
    "d_cpt":             run_d_cpt,
    "d_icd_procedures":  run_d_icd_procedures,
}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    spark = build_spark(args.master, args.spark_temp_dir)

    print("=" * 65)
    print("  nxmanh – MIMIC-III Preprocessing Pipeline")
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
