"""
preprocessing.py – PROCEDUREEVENTS_MV Pipeline (Bronze → Silver → Gold)

Đặc điểm bảng này:
  - Silver: tự động loại cột có >50% Null (dynamic), không hardcode.
  - Duplicate check: Self-Join với tolerance thời gian 1h.
  - Gold: Deduplication bằng Window ROW_NUMBER (ưu tiên 'FinishedRunning').
  - Verification: So sánh Silver vs Gold trên một HADM_ID mẫu.

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
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TABLE_NAME  = "PROCEDUREEVENTS_MV"
INPUT_PATH  = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/PROCEDUREEVENTS_MV.csv.gz"
BRONZE_PATH = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/bronze_procedureevents_mv.parquet"
SILVER_PATH = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/silver_procedureevents_mv.parquet"
GOLD_PATH   = r"C:/Users/nguye/Downloads/MIMIC-III_Dataset/processed/gold_procedureevents_mv.parquet"

# Khóa chính bắt buộc có giá trị ở bước Silver
PRIMARY_KEYS      = ["HADM_ID", "STARTTIME", "ITEMID"]
# Ngưỡng tỷ lệ Null để tự động loại cột (50%)
NULL_DROP_THRESHOLD = 0.5
# Số partition khi ghi Gold
GOLD_PARTITIONS   = 50
# HADM_ID mẫu dùng để verify kết quả Gold
SAMPLE_HADM_ID    = 100039


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
# BƯỚC 2 – KIỂM TRA NULL + PHÂN PHỐI HADM_ID (báo cáo, không thay đổi data)
# ---------------------------------------------------------------------------
def report_null_counts(df: DataFrame) -> None:
    """
    In báo cáo tỷ lệ Null của tất cả các cột trong một Spark Job duy nhất.

    Args:
        df: DataFrame cần kiểm tra.
    """
    total_count = df.count()
    print(f"\n[NULL CHECK] Tổng số bản ghi: {total_count:,}")

    null_exprs = [
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]
    null_counts_row = df.select(*null_exprs).collect()[0].asDict()

    separator = "-" * 65
    print(separator)
    print(f"{'TÊN CỘT':<30} | {'SỐ DÒNG NULL':<15} | TỶ LỆ %")
    print(separator)

    for col_name in df.columns:
        null_val = null_counts_row[col_name]
        null_pct = (null_val / total_count * 100) if total_count > 0 else 0.0
        print(f"{col_name:<30} | {null_val:<15} | {null_pct:.2f}%")

    print(separator)


def report_bucket_distribution(df: DataFrame) -> None:
    """
    Kiểm tra phân phối blocking theo HADM_ID:
      - Số bucket có count > 1 (tiềm năng trùng lặp).
      - Thống kê avg/max/min/stddev.
      - Top 10 bucket nhiều bản ghi nhất.

    Args:
        df: DataFrame Silver cần kiểm tra.
    """
    bucket_dist = df.groupBy("HADM_ID").count()

    # Thống kê tổng quát
    stats = bucket_dist.select(
        F.mean("count").alias("avg_rows"),
        F.max("count").alias("max_rows"),
        F.min("count").alias("min_rows"),
        F.stddev("count").alias("stddev_rows"),
    ).collect()[0]

    potential_dup_count = bucket_dist.filter("count > 1").count()

    separator = "-" * 60
    print(f"\n{separator}")
    print("THỐNG KÊ PHÂN BỔ BUCKET (HADM_ID)")
    print(separator)
    print(f"Số bucket có count > 1 (tiềm năng trùng): {potential_dup_count:,}")
    print(f"Trung bình mỗi bucket : {stats['avg_rows']:.2f} rows")
    print(f"Bucket lớn nhất       : {stats['max_rows']} rows")
    print(f"Bucket nhỏ nhất       : {stats['min_rows']} rows")
    print(f"Độ lệch chuẩn         : {stats['stddev_rows']:.2f}")
    print(f"\n[Top 10 bucket nhiều bản ghi nhất]")
    bucket_dist.orderBy(F.desc("count")).show(10, truncate=False)


# ---------------------------------------------------------------------------
# BƯỚC 3 – TẠO LỚP SILVER
# ---------------------------------------------------------------------------
def create_silver(spark: SparkSession) -> DataFrame:
    """
    Làm sạch dữ liệu Bronze:
      - Tự động phát hiện và loại cột có tỷ lệ Null > NULL_DROP_THRESHOLD.
      - Loại dòng thiếu khóa chính (PRIMARY_KEYS).
    Lưu kết quả thành lớp Silver.

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Silver đã làm sạch.
    """
    df_bronze  = spark.read.parquet(BRONZE_PATH)
    total_rows = df_bronze.count()

    # Tính null ratio cho từng cột trong 1 job
    null_exprs = [
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df_bronze.columns
    ]
    null_counts = df_bronze.select(*null_exprs).collect()[0].asDict()

    cols_to_drop = [
        col_name for col_name, null_val in null_counts.items()
        if total_rows > 0 and (null_val / total_rows) > NULL_DROP_THRESHOLD
    ]
    print(f"[SILVER] Cột tự động loại bỏ (>{NULL_DROP_THRESHOLD*100:.0f}% Null): {cols_to_drop}")

    df_silver = df_bronze.drop(*cols_to_drop).dropna(subset=PRIMARY_KEYS)
    df_silver.write.mode("overwrite").parquet(SILVER_PATH)
    print(f"[SILVER] ✓ Đã tạo lớp Silver – {len(df_silver.columns)} cột còn lại, "
          f"{df_silver.count():,} dòng.")
    return df_silver


# ---------------------------------------------------------------------------
# BƯỚC 4 – PHÂN TÍCH TRÙNG LẶP BẰNG SELF-JOIN (báo cáo, không thay đổi data)
# ---------------------------------------------------------------------------
def analyze_duplicate_pairs(spark: SparkSession) -> None:
    """
    Self-Join để tìm các cặp bản ghi nghi trùng lặp theo tiêu chí:
      - Cùng HADM_ID, SUBJECT_ID, ITEMID, VALUE, CGID.
      - Chênh lệch STARTTIME < 1h VÀ ENDTIME < 1h.
    Hiển thị tối đa 20 cặp đầu tiên.

    Args:
        spark: SparkSession đang chạy.
    """
    df = spark.read.parquet(SILVER_PATH)
    df_a = df.alias("a")
    df_b = df.alias("b")

    condition = [
        F.col("a.HADM_ID")    == F.col("b.HADM_ID"),
        F.col("a.SUBJECT_ID") == F.col("b.SUBJECT_ID"),
        F.col("a.ROW_ID")     <  F.col("b.ROW_ID"),   # Tránh cặp đảo ngược
        F.col("a.ITEMID")     == F.col("b.ITEMID"),
        F.col("a.VALUE")      == F.col("b.VALUE"),
        F.col("a.CGID")       == F.col("b.CGID"),
    ]

    start_diff = F.abs(F.unix_timestamp("a.STARTTIME") - F.unix_timestamp("b.STARTTIME"))
    end_diff   = F.abs(F.unix_timestamp("a.ENDTIME")   - F.unix_timestamp("b.ENDTIME"))

    result = (
        df_a.join(df_b, condition)
        .filter((start_diff < 3600) & (end_diff < 3600))
        .select(
            F.col("a.HADM_ID"),
            F.col("a.ITEMID"),
            F.col("a.CGID"),
            F.col("a.STATUSDESCRIPTION").alias("STATUS"),
            F.col("a.ROW_ID").alias("ID_1"),
            F.col("a.STARTTIME").alias("START_1"),
            F.col("a.ENDTIME").alias("END_1"),
            F.col("a.VALUE").alias("VAL_1"),
            F.col("b.ROW_ID").alias("ID_2"),
            F.col("b.STARTTIME").alias("START_2"),
            F.col("b.ENDTIME").alias("END_2"),
            F.col("b.VALUE").alias("VAL_2"),
        )
    )

    cnt = result.count()
    separator = "-" * 80
    print(f"\n{separator}")
    print("PHÂN TÍCH ĐỐI SOÁT CHI TIẾT (SELF-JOIN)")
    print(f"Số cặp trùng (khớp ITEMID, VALUE, CGID, TIME ±1h): {cnt:,}")
    print(separator)

    if cnt > 0:
        result.orderBy("a.HADM_ID", "a.STARTTIME").show(20, truncate=False)
    else:
        print("Không tìm thấy bản ghi nào khớp với các tiêu chí.")


# ---------------------------------------------------------------------------
# BƯỚC 5 – TẠO LỚP GOLD (DEDUPLICATION)
# ---------------------------------------------------------------------------
def create_gold(spark: SparkSession) -> DataFrame:
    """
    Loại bỏ trùng lặp (Deduplication) bằng Window ROW_NUMBER:
      - Partition theo (HADM_ID, ITEMID, VALUE, CGID, STARTTIME, ENDTIME).
      - Ưu tiên giữ dòng 'FinishedRunning' trước, sau đó theo ROW_ID tăng dần.
    Lưu kết quả thành lớp Gold.

    Args:
        spark: SparkSession đang chạy.

    Returns:
        DataFrame Gold đã dedup.
    """
    df            = spark.read.parquet(SILVER_PATH)
    initial_count = df.count()

    # Cột tạm: FinishedRunning = 0 (ưu tiên cao nhất), còn lại = 1
    df_with_priority = df.withColumn(
        "priority",
        F.when(F.col("STATUSDESCRIPTION") == "FinishedRunning", 0).otherwise(1),
    )

    window_spec = Window.partitionBy(
        "HADM_ID", "ITEMID", "VALUE", "CGID", "STARTTIME", "ENDTIME"
    ).orderBy("priority", "ROW_ID")

    df_gold = (
        df_with_priority
        .withColumn("rank", F.row_number().over(window_spec))
        .filter("rank == 1")
        .drop("priority", "rank")
    )

    df_gold.repartition(GOLD_PARTITIONS).write.mode("overwrite").parquet(GOLD_PATH)

    final_count   = df_gold.count()
    removed_count = initial_count - final_count

    separator = "-" * 60
    print(f"\n{separator}")
    print(f"KẾT QUẢ DEDUPLICATION – LỚP GOLD {TABLE_NAME}")
    print(separator)
    print(f"Tổng số dòng Silver ban đầu : {initial_count:,}")
    print(f"Số dòng Gold sau dedup      : {final_count:,}")
    print(f"Số dòng trùng đã loại bỏ   : {removed_count:,} "
          f"({removed_count/initial_count*100:.2f}%)" if initial_count > 0 else "")
    print(separator)

    return df_gold


# ---------------------------------------------------------------------------
# BƯỚC 6 – KIỂM TRA KẾT QUẢ (so sánh Silver vs Gold trên 1 HADM_ID mẫu)
# ---------------------------------------------------------------------------
def verify_gold(spark: SparkSession, target_hadm: int = SAMPLE_HADM_ID) -> None:
    """
    So sánh dữ liệu Silver và Gold cho một HADM_ID cụ thể để xác nhận
    deduplication hoạt động đúng.

    Args:
        spark       : SparkSession đang chạy.
        target_hadm : HADM_ID cần kiểm tra (mặc định = SAMPLE_HADM_ID).
    """
    df_silver = spark.read.parquet(SILVER_PATH)
    df_gold   = spark.read.parquet(GOLD_PATH)

    cols_show = ["ROW_ID", "ITEMID", "STARTTIME", "ENDTIME", "VALUE", "STATUSDESCRIPTION"]

    print(f"\n[VERIFY] Kiểm tra HADM_ID = {target_hadm}")

    print("\n[SILVER – Trước khi dedup]")
    df_silver.filter(F.col("HADM_ID") == target_hadm) \
        .select(*cols_show).orderBy("STARTTIME", "ROW_ID").show(truncate=False)

    print("\n[GOLD – Sau khi dedup]")
    df_gold.filter(F.col("HADM_ID") == target_hadm) \
        .select(*cols_show).orderBy("STARTTIME", "ROW_ID").show(truncate=False)

    count_before = df_silver.filter(F.col("HADM_ID") == target_hadm).count()
    count_after  = df_gold.filter(F.col("HADM_ID")   == target_hadm).count()
    print(f"Số dòng ban đầu : {count_before}")
    print(f"Số dòng sau dedup: {count_after}")
    print(f"Đã loại bỏ      : {count_before - count_after} dòng")


# ---------------------------------------------------------------------------
# MAIN – Điều phối toàn bộ pipeline
# ---------------------------------------------------------------------------
def main(spark: SparkSession) -> None:
    """
    Luồng xử lý chính:
      Bronze → Null Check → Bucket Stats → Silver → Self-Join Analysis → Gold → Verify.

    Args:
        spark: SparkSession được truyền vào (từ môi trường Zeppelin/spark-submit).
    """
    print("=" * 65)
    print(f"  PIPELINE {TABLE_NAME}")
    print("=" * 65)

    # Bước 1: Tạo Bronze
    df_bronze = create_bronze(spark)

    # Bước 2a: Báo cáo Null trên Bronze
    report_null_counts(df_bronze)

    # Bước 2b: Phân phối bucket trên Bronze (để định hướng Silver)
    report_bucket_distribution(df_bronze)

    # Bước 3: Tạo Silver (auto-drop >50% null, dropna key cols)
    create_silver(spark)

    # Bước 4: Phân tích cặp trùng lặp chi tiết trên Silver (self-join)
    analyze_duplicate_pairs(spark)

    # Bước 5: Tạo Gold (deduplication)
    create_gold(spark)

    # Bước 6: Kiểm tra kết quả
    verify_gold(spark)

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