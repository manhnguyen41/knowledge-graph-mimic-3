"""
preprocessing.py – PROCEDURES_ICD Pipeline (Bronze → Silver → Gold)

Chạy bằng: python preprocessing.py
           hoặc import vào notebook và gọi main(spark).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# WINDOWS: Set HADOOP_HOME trước khi import PySpark
# ---------------------------------------------------------------------------
import os
import sys

_HADOOP_HOME = r"C:\hadoop"
if sys.platform == "win32":
    os.environ.setdefault("HADOOP_HOME", _HADOOP_HOME)
    os.environ.setdefault("hadoop.home.dir", _HADOOP_HOME)
    _bin = os.path.join(_HADOOP_HOME, "bin")
    if _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
    # Fix UnicodeEncodeError trên Windows terminal (CP1252)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TABLE_NAME  = "PROCEDURES_ICD"
INPUT_PATH  = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/PROCEDURES_ICD.csv.gz"
BRONZE_PATH = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/bronze_procedures_icd.parquet"
SILVER_PATH = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/silver_procedures_icd.parquet"
GOLD_PATH   = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/gold_procedures_icd.parquet"

# Khóa chính bắt buộc phải có giá trị
PRIMARY_KEYS    = ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]
# Số partition khi ghi Gold
GOLD_PARTITIONS = 50


# ---------------------------------------------------------------------------
# BƯỚC 1 – TẠO LỚP BRONZE
# ---------------------------------------------------------------------------
def create_bronze(spark: SparkSession) -> DataFrame:
    """
    Đọc file CSV thô và lưu sang Parquet (lớp Bronze).

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Bronze chưa qua xử lý.
    """
    df_raw = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
    df_raw.write.mode("overwrite").parquet(BRONZE_PATH)
    print(f"[BRONZE] ✓ Đã tạo lớp Bronze tại {BRONZE_PATH}")
    return df_raw


# ---------------------------------------------------------------------------
# BƯỚC 2 – KIỂM TRA NULL (báo cáo, không thay đổi dữ liệu)
# ---------------------------------------------------------------------------
def report_null_counts(df: DataFrame) -> None:
    """
    In báo cáo tỷ lệ Null của tất cả các cột trong một Spark Job duy nhất.

    Args:
        df: DataFrame cần kiểm tra.
    """
    total_count = df.count()
    print(f"\n[NULL CHECK] Đang quét toàn bộ {total_count:,} dòng dữ liệu...")

    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    null_counts_row = df.select(*null_exprs).collect()[0]

    separator = "-" * 65
    print(separator)
    print(f"{'TÊN CỘT':<25} | {'SỐ DÒNG NULL':<15} | TỶ LỆ %")
    print(separator)

    for col_name in df.columns:
        null_count = null_counts_row[col_name]
        null_pct   = (null_count / total_count * 100) if total_count > 0 else 0.0
        print(f"{col_name:<25} | {null_count:<15} | {null_pct:.2f}%")

    print(separator)


# ---------------------------------------------------------------------------
# BƯỚC 3 – TẠO LỚP SILVER
# ---------------------------------------------------------------------------
def create_silver(spark: SparkSession) -> DataFrame:
    """
    Làm sạch dữ liệu Bronze:
      - Loại dòng thiếu khóa chính (PRIMARY_KEYS).
    Lưu kết quả thành lớp Silver.

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Silver đã làm sạch.
    """
    df_bronze = spark.read.parquet(BRONZE_PATH)
    df_silver = df_bronze.dropna(subset=PRIMARY_KEYS)
    df_silver.write.mode("overwrite").parquet(SILVER_PATH)
    print(f"[SILVER] ✓ Lớp Silver đã sàng lọc xong. Số dòng: {df_silver.count():,}")
    return df_silver


# ---------------------------------------------------------------------------
# BƯỚC 4 – KIỂM TRA TRÙNG LẶP (báo cáo, không thay đổi dữ liệu)
# ---------------------------------------------------------------------------
def report_duplicates(spark: SparkSession) -> None:
    """
    Dùng GroupBy để phát hiện các tổ hợp (HADM_ID, ICD9_CODE) xuất hiện
    nhiều hơn 1 lần. Hiển thị mẫu top-5 nếu có trùng lặp.

    Args:
        spark: SparkSession đang chạy.
    """
    df_silver = spark.read.parquet(SILVER_PATH)

    df_check_dup = df_silver.groupBy("HADM_ID", "ICD9_CODE").count()
    dup_cases    = df_check_dup.filter("count > 1").count()

    separator = "-" * 60
    print(f"\n{separator}")
    print(f"KIỂM TRA TRÙNG LẶP ({TABLE_NAME})")
    print(separator)
    print(f"Số tổ hợp (Nhập viện + Thủ thuật) bị lặp lại: {dup_cases:,}")
    print(separator)

    if dup_cases > 0:
        print("\n[Mẫu các ca trùng lặp – top 5]")
        df_check_dup.filter("count > 1").orderBy(F.desc("count")).show(5, truncate=False)


# ---------------------------------------------------------------------------
# BƯỚC 5 – TẠO LỚP GOLD
# ---------------------------------------------------------------------------
def create_gold(spark: SparkSession) -> DataFrame:
    """
    Tổng hợp lớp Silver thành lớp Gold bằng Group By + Aggregation:
      - Gom nhóm theo (SUBJECT_ID, HADM_ID, ICD9_CODE).
      - PROC_COUNT     : số lần thủ thuật được ghi nhận.
      - PRIMARY_SEQ_NUM: SEQ_NUM nhỏ nhất làm vị trí đại diện.
    Lưu kết quả với GOLD_PARTITIONS partition.

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Gold đã lưu.
    """
    df_silver = spark.read.parquet(SILVER_PATH)

    df_gold = (
        df_silver
        .groupBy("SUBJECT_ID", "HADM_ID", "ICD9_CODE")
        .agg(
            F.count("*").alias("PROC_COUNT"),
            F.min("SEQ_NUM").alias("PRIMARY_SEQ_NUM"),
        )
    )

    df_gold_final = df_gold.repartition(GOLD_PARTITIONS)
    df_gold_final.write.mode("overwrite").parquet(GOLD_PATH)

    edge_count = df_gold_final.count()
    separator  = "-" * 60
    print(f"\n{separator}")
    print(f"XỬ LÝ THÀNH CÔNG: LỚP GOLD {TABLE_NAME}")
    print(separator)
    print(f"[GOLD] ✓ Số cạnh (Edge) thủ thuật ICD-9 trên Graph: {edge_count:,}")

    return df_gold_final


# ---------------------------------------------------------------------------
# MAIN – Điều phối toàn bộ pipeline
# ---------------------------------------------------------------------------
def main(spark: SparkSession) -> None:
    """
    Luồng xử lý chính: Bronze → Null Check → Silver → Duplicate Check → Gold.

    Args:
        spark: SparkSession được truyền vào (từ môi trường Zeppelin/spark-submit).
    """
    print("=" * 65)
    print(f"  PIPELINE {TABLE_NAME}")
    print("=" * 65)

    # Bước 1: Tạo Bronze
    df_bronze = create_bronze(spark)

    # Bước 2: Báo cáo Null trên Bronze
    report_null_counts(df_bronze)

    # Bước 3: Tạo Silver
    create_silver(spark)

    # Bước 4: Báo cáo trùng lặp trên Silver
    report_duplicates(spark)

    # Bước 5: Tạo Gold
    create_gold(spark)

    print("\n" + "=" * 65)
    print("  PIPELINE HOÀN THÀNH!")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point khi chạy bằng: python preprocessing.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _spark = (
        SparkSession.builder
        .appName(f"preprocessing_{TABLE_NAME}")
        .getOrCreate()
    )
    main(_spark)
    _spark.stop()