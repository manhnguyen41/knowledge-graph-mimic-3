"""
preprocessing.py – CPTEVENTS Pipeline (Bronze → Silver → Gold)

Chuyển đổi từ Zeppelin notebook sang Python module hiện đại.
Chạy bằng: python preprocessing.py
           hoặc import vào notebook và gọi main(spark).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# WINDOWS: Set HADOOP_HOME trước khi import PySpark
# winutils.exe cần thiết để Spark ghi file local trên Windows.
# Tải tại: https://github.com/cdarlint/winutils
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

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TABLE_NAME   = "CPTEVENTS"
INPUT_PATH   = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/CPTEVENTS.csv.gz"
BRONZE_PATH  = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/bronze_cptevents.parquet"
SILVER_PATH  = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/silver_cptevents.parquet"
GOLD_PATH    = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/gold_cptevents.parquet"

# Cột loại bỏ ở bước Silver: CPT_SUFFIX (100% null), DESCRIPTION (82% null)
COLS_TO_DROP  = ["CPT_SUFFIX", "DESCRIPTION"]
# Khóa chính bắt buộc phải có giá trị
PRIMARY_KEYS  = ["SUBJECT_ID", "HADM_ID", "CPT_CD"]
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
      - Xóa các cột rác (COLS_TO_DROP).
      - Loại dòng thiếu khóa chính (PRIMARY_KEYS).
    Lưu kết quả thành lớp Silver.

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Silver đã làm sạch.
    """
    df_bronze = spark.read.parquet(BRONZE_PATH)
    df_silver = df_bronze.drop(*COLS_TO_DROP).dropna(subset=PRIMARY_KEYS)
    df_silver.write.mode("overwrite").parquet(SILVER_PATH)
    print(f"[SILVER] ✓ Đã tạo lớp Silver – bỏ cột: {COLS_TO_DROP}")
    return df_silver


# ---------------------------------------------------------------------------
# BƯỚC 4 – KIỂM TRA TRÙNG LẶP (báo cáo, không thay đổi dữ liệu)
# ---------------------------------------------------------------------------
def report_duplicates(spark: SparkSession) -> None:
    """
    Dùng Window Function (ROW_NUMBER) để phát hiện và báo cáo
    các dòng trùng lặp theo tổ hợp (HADM_ID, CPT_CD).
    Hiển thị mẫu minh họa nếu có trùng lặp.

    Args:
        spark: SparkSession đang chạy.
    """
    df_silver   = spark.read.parquet(SILVER_PATH)
    silver_count = df_silver.count()

    window_spec = Window.partitionBy("HADM_ID", "CPT_CD").orderBy("ROW_ID")
    df_ranked   = df_silver.withColumn("rank", F.row_number().over(window_spec))

    dup_count    = df_ranked.filter("rank > 1").count()
    unique_count = silver_count - dup_count
    dup_pct      = (dup_count / silver_count * 100) if silver_count > 0 else 0.0

    separator = "-" * 60
    print(f"\n{separator}")
    print("BÁO CÁO KIỂM TRA TRÙNG LẶP (CPTEVENTS)")
    print(separator)
    print(f"Tổng số dòng Silver:              {silver_count:,}")
    print(f"Số cạnh (Edge) duy nhất:          {unique_count:,}")
    print(f"Số dòng lặp sẽ bị xóa:            {dup_count:,} ({dup_pct:.2f}%)")
    print(separator)

    if dup_count > 0:
        print("\n[Mẫu so sánh: rank=1 (GIỮ) vs rank>1 (XÓA)]")
        df_ranked.createOrReplaceTempView("ranked_cpt")

        sample = spark.sql(
            "SELECT HADM_ID, CPT_CD FROM ranked_cpt WHERE rank = 2 LIMIT 1"
        ).collect()

        if sample:
            s_hadm = sample[0]["HADM_ID"]
            s_cpt  = sample[0]["CPT_CD"]
            spark.sql(f"""
                SELECT rank, HADM_ID, CPT_CD, SECTIONHEADER, SUBSECTIONHEADER, ROW_ID
                FROM ranked_cpt
                WHERE HADM_ID = {s_hadm} AND CPT_CD = '{s_cpt}'
                ORDER BY rank
            """).show(truncate=False)


# ---------------------------------------------------------------------------
# BƯỚC 5 – TẠO LỚP GOLD
# ---------------------------------------------------------------------------
def create_gold(spark: SparkSession) -> DataFrame:
    """
    Tổng hợp lớp Silver thành lớp Gold bằng Group By + Count:
      - Gom nhóm theo (SUBJECT_ID, HADM_ID, CPT_CD, SECTIONHEADER, SUBSECTIONHEADER).
      - Thêm cột TICKET_COUNT = số lần thủ thuật được ghi nhận.
    Lưu kết quả với {GOLD_PARTITIONS} partition.

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Gold đã lưu.
    """
    df_silver = spark.read.parquet(SILVER_PATH)

    df_gold = (
        df_silver
        .groupBy("SUBJECT_ID", "HADM_ID", "CPT_CD", "SECTIONHEADER", "SUBSECTIONHEADER")
        .agg(F.count("*").alias("TICKET_COUNT"))
    )

    df_gold_final = df_gold.repartition(GOLD_PARTITIONS)
    df_gold_final.write.mode("overwrite").parquet(GOLD_PATH)

    edge_count = df_gold_final.count()
    separator  = "-" * 60
    print(f"\n{separator}")
    print("XỬ LÝ THÀNH CÔNG: LỚP GOLD CPTEVENTS (CÓ TẦN SUẤT)")
    print(separator)
    print(f"[GOLD] ✓ Số cạnh (Edge) trên Graph: {edge_count:,}")

    # Mẫu kiểm tra
    print("\n[Kết quả mẫu HADM_ID=100039, CPT_CD='99254']")
    df_gold_final.filter(
        (F.col("HADM_ID") == 100039) & (F.col("CPT_CD") == "99254")
    ).show(truncate=False)

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
    print(f"  PIPELINE CPTEVENTS – Bảng: {TABLE_NAME}")
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
# Entry point khi chạy bằng spark-submit
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _spark = (
        SparkSession.builder
        .appName(f"preprocessing_{TABLE_NAME}")
        .getOrCreate()
    )
    main(_spark)
    _spark.stop()