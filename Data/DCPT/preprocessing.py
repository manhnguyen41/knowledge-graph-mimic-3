"""
preprocessing.py – D_CPT Pipeline (Bronze → Gold)

Bảng từ điển tham chiếu (Dimension Table) – không có bước Silver riêng
vì không cần làm sạch phức tạp. Pipeline gồm:
  1. Bronze : Đọc CSV thô → Parquet.
  2. Null Check : Báo cáo chất lượng, highlight cột có null.
  3. Sample View: Xem cấu trúc phân cấp CPT.
  4. Gold : Loại dòng thiếu khoá chính → Parquet sẵn cho Graph.

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
TABLE_NAME  = "D_CPT"
INPUT_PATH  = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/D_CPT.csv.gz"
BRONZE_PATH = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/bronze_d_cpt.parquet"
GOLD_PATH   = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/gold_d_cpt.parquet"

# Khóa chính: dòng thiếu các cột này sẽ bị loại ở bước Gold
PRIMARY_KEYS = ["CATEGORY", "MINCODEINSUBSECTION", "MAXCODEINSUBSECTION"]
# Cột hiển thị cấu trúc phân cấp CPT
HIERARCHY_COLS = [
    "CATEGORY", "SECTIONHEADER", "SUBSECTIONHEADER",
    "MINCODEINSUBSECTION", "MAXCODEINSUBSECTION",
]
GOLD_PARTITIONS = 10  # Bảng từ điển nhỏ, ít partition là đủ


# ---------------------------------------------------------------------------
# BƯỚC 1 – TẠO LỚP BRONZE
# ---------------------------------------------------------------------------
def create_bronze(spark: SparkSession) -> DataFrame:
    """
    Đọc file CSV thô D_CPT và lưu sang Parquet (lớp Bronze).

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
    print(f"\n[NULL CHECK] Đang quét toàn bộ {total_count:,} dòng của bảng từ điển {TABLE_NAME}...")

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
        annotation = "  <-- Chú ý" if null_count > 0 else ""
        print(f"{col_name:<25} | {null_count:<15} | {null_pct:.2f}%{annotation}")

    print(separator)


# ---------------------------------------------------------------------------
# BƯỚC 3 – XEM CẤU TRÚC PHÂN CẤP CPT (báo cáo, không thay đổi dữ liệu)
# ---------------------------------------------------------------------------
def show_hierarchy_sample(df: DataFrame, n: int = 10) -> None:
    """
    Hiển thị mẫu cấu trúc phân cấp của bảng D_CPT
    (CATEGORY → SECTIONHEADER → SUBSECTIONHEADER → code range).

    Args:
        df: DataFrame Bronze hoặc Gold.
        n : Số dòng hiển thị (mặc định 10).
    """
    print(f"\n[MẪU] Cấu trúc phân cấp CPT (top {n} dòng theo MINCODEINSUBSECTION):")
    df.select(*HIERARCHY_COLS).orderBy("MINCODEINSUBSECTION").show(n, truncate=False)


# ---------------------------------------------------------------------------
# BƯỚC 4 – TẠO LỚP GOLD
# ---------------------------------------------------------------------------
def create_gold(spark: SparkSession) -> DataFrame:
    """
    Làm sạch Bronze thành Gold:
      - Loại dòng thiếu khoá chính (PRIMARY_KEYS).
    Bảng từ điển nhỏ nên dùng ít partition hơn.

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Gold sẵn sàng cho Graph.
    """
    df_bronze = spark.read.parquet(BRONZE_PATH)
    df_gold   = df_bronze.dropna(subset=PRIMARY_KEYS)
    df_gold.repartition(GOLD_PARTITIONS).write.mode("overwrite").parquet(GOLD_PATH)

    gold_count = df_gold.count()
    separator  = "-" * 65
    print(f"\n{separator}")
    print(f"XỬ LÝ THÀNH CÔNG: LỚP GOLD {TABLE_NAME}")
    print(separator)
    print(f"[GOLD] ✓ Số bản ghi từ điển: {gold_count:,}")

    return df_gold


# ---------------------------------------------------------------------------
# MAIN – Điều phối toàn bộ pipeline
# ---------------------------------------------------------------------------
def main(spark: SparkSession) -> None:
    """
    Luồng xử lý chính: Bronze → Null Check → Hierarchy Sample → Gold.

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

    # Bước 3: Xem cấu trúc phân cấp
    show_hierarchy_sample(df_bronze)

    # Bước 4: Tạo Gold
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