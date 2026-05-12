"""
preprocessing.py – D_ICD_PROCEDURES Pipeline (Bronze → Gold)

Bảng từ điển tham chiếu (Dimension Table) cho mã thủ thuật ICD-9.
Pipeline gồm:
  1. Bronze    : Đọc CSV thô → Parquet.
  2. Null Check: Báo cáo chất lượng, highlight cột có null.
  3. Gold      : Loại dòng thiếu ICD9_CODE → Parquet sẵn cho Graph.
  4. Sample    : Xem mẫu ICD9_CODE / SHORT_TITLE / LONG_TITLE.

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
TABLE_NAME  = "D_ICD_PROCEDURES"
INPUT_PATH  = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/D_ICD_PROCEDURES.csv.gz"
BRONZE_PATH = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/bronze_d_icd_procedures.parquet"
GOLD_PATH   = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/gold_d_icd_procedures.parquet"

# Khoá chính: dòng thiếu cột này sẽ bị loại ở bước Gold
PRIMARY_KEYS    = ["ICD9_CODE"]
# Cột hiển thị mẫu dữ liệu
SAMPLE_COLS     = ["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"]
GOLD_PARTITIONS = 10   # Bảng từ điển nhỏ, ít partition là đủ


# ---------------------------------------------------------------------------
# BƯỚC 1 – TẠO LỚP BRONZE
# ---------------------------------------------------------------------------
def create_bronze(spark: SparkSession) -> DataFrame:
    """
    Đọc file CSV thô D_ICD_PROCEDURES và lưu sang Parquet (lớp Bronze).

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
# BƯỚC 2 – KIỂM TRA NULL (báo cáo, highlight cột có null)
# ---------------------------------------------------------------------------
def report_null_counts(df: DataFrame) -> None:
    """
    In báo cáo tỷ lệ Null cho tất cả cột.
    Các cột có Null > 0 sẽ được đánh dấu '<-- Chú ý'.

    Args:
        df: DataFrame cần kiểm tra.
    """
    total_count = df.count()
    print(f"\n[NULL CHECK] Đang quét toàn bộ {total_count:,} dòng của từ điển {TABLE_NAME}...")

    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    null_counts_row = df.select(*null_exprs).collect()[0]

    separator = "-" * 65
    print(separator)
    print(f"{'TÊN CỘT':<20} | {'SỐ DÒNG NULL':<15} | TỶ LỆ %")
    print(separator)

    for col_name in df.columns:
        null_count = null_counts_row[col_name]
        null_pct   = (null_count / total_count * 100) if total_count > 0 else 0.0
        annotation = "  <-- Chú ý" if null_count > 0 else ""
        print(f"{col_name:<20} | {null_count:<15} | {null_pct:.2f}%{annotation}")

    print(separator)


# ---------------------------------------------------------------------------
# BƯỚC 3 – TẠO LỚP GOLD
# ---------------------------------------------------------------------------
def create_gold(spark: SparkSession) -> DataFrame:
    """
    Làm sạch Bronze thành Gold:
      - Loại dòng thiếu ICD9_CODE (khoá chính của từ điển).
    Lưu kết quả Parquet với ít partition vì bảng nhỏ.

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Gold sẵn sàng cho Graph.
    """
    df_bronze = spark.read.parquet(BRONZE_PATH)
    df_gold   = df_bronze.dropna(subset=PRIMARY_KEYS)
    df_gold.repartition(GOLD_PARTITIONS).write.mode("overwrite").parquet(GOLD_PATH)

    separator = "-" * 65
    print(f"\n{separator}")
    print(f"XỬ LÝ THÀNH CÔNG: LỚP GOLD {TABLE_NAME}")
    print(separator)
    print(f"[GOLD] ✓ Số bản ghi từ điển ICD-9: {df_gold.count():,}")

    return df_gold


# ---------------------------------------------------------------------------
# BƯỚC 4 – HIỂN THỊ MẪU (báo cáo, không thay đổi dữ liệu)
# ---------------------------------------------------------------------------
def show_sample(df: DataFrame, n: int = 10) -> None:
    """
    Hiển thị mẫu dữ liệu từ điển ICD-9 (mã, tên ngắn, tên đầy đủ).

    Args:
        df: DataFrame Gold cần hiển thị.
        n : Số dòng hiển thị (mặc định 10).
    """
    print(f"\n[MẪU] Từ điển thủ thuật ICD-9 (top {n} dòng):")
    df.select(*SAMPLE_COLS).show(n, truncate=False)


# ---------------------------------------------------------------------------
# MAIN – Điều phối toàn bộ pipeline
# ---------------------------------------------------------------------------
def main(spark: SparkSession) -> None:
    """
    Luồng xử lý chính: Bronze → Null Check → Gold → Sample.

    Args:
        spark: SparkSession được truyền vào.
    """
    print("=" * 65)
    print(f"  PIPELINE {TABLE_NAME} (Dimension Table)")
    print("=" * 65)

    # Bước 1: Tạo Bronze
    df_bronze = create_bronze(spark)

    # Bước 2: Báo cáo Null
    report_null_counts(df_bronze)

    # Bước 3: Tạo Gold
    df_gold = create_gold(spark)

    # Bước 4: Hiển thị mẫu
    show_sample(df_gold)

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