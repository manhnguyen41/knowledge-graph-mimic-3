```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

# ============================================================================
# CONFIG
# ============================================================================

# -----------------------------
# INPUT PATHS
# -----------------------------

INPUT_CV_PATH = "/tmp/INPUTEVENTS_CV.csv.gz"
INPUT_MV_PATH = "/tmp/INPUTEVENTS_MV.csv.gz"

# -----------------------------
# BRONZE PATHS
# -----------------------------

BRONZE_CV_PATH = "/tmp/bronze_inputevents_cv"
BRONZE_MV_PATH = "/tmp/bronze_inputevents_mv"

# -----------------------------
# SILVER PATHS
# -----------------------------

SILVER_CV_PATH = "/tmp/silver_inputevents_cv"
SILVER_MV_PATH = "/tmp/silver_inputevents_mv"

# -----------------------------
# GOLD PATHS
# -----------------------------

GOLD_CV_PATH = "/tmp/gold_inputevents_cv"
GOLD_MV_PATH = "/tmp/gold_inputevents_mv"

# -----------------------------
# FINAL FUSION PATH
# -----------------------------

FUSION_GOLD_PATH = "/tmp/gold_unified_inputevents"

# ============================================================================
# SPARK SESSION
# ============================================================================

spark = (
    SparkSession.builder
    .appName("MIMICIII_INPUTEVENTS_PIPELINE")
    .getOrCreate()
)

# ============================================================================
# COMMON UTILITIES
# ============================================================================

def print_separator(title: str):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def null_report(df: DataFrame):

    print_separator("NULL REPORT")

    total_rows = df.count()

    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])

    null_counts.show(truncate=False)

    print(f"Total rows: {total_rows:,}")


# ============================================================================
# PART A — INPUTEVENTS_CV
# ============================================================================

def create_cv_pipeline(spark: SparkSession):

    # ------------------------------------------------------------------------
    # LOAD RAW CSV
    # ------------------------------------------------------------------------

    input_cv_raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(INPUT_CV_PATH)
    )

    print_separator("INPUTEVENTS_CV LOADED")

    print(f"Raw CV rows: {input_cv_raw.count():,}")

    input_cv_raw.printSchema()

    input_cv_raw.show(5, truncate=False)

    # ------------------------------------------------------------------------
    # NULL REPORT
    # ------------------------------------------------------------------------

    null_report(input_cv_raw)

    # ------------------------------------------------------------------------
    # CREATE BRONZE
    # ------------------------------------------------------------------------

    (
        input_cv_raw.write
        .mode("overwrite")
        .parquet(BRONZE_CV_PATH)
    )

    print_separator("BRONZE CV CREATED")

    bronze_cv = spark.read.parquet(BRONZE_CV_PATH)

    # ------------------------------------------------------------------------
    # CREATE SILVER
    # ------------------------------------------------------------------------

    silver_cv = (

        bronze_cv

        # Remove duplicates
        .dropDuplicates()

        # Remove invalid rows
        .filter(
            F.col("SUBJECT_ID").isNotNull() &
            F.col("HADM_ID").isNotNull() &
            F.col("ITEMID").isNotNull()
        )

        # Harmonize event time
        .withColumnRenamed(
            "CHARTTIME",
            "EVENT_TIME"
        )

        # Add END_TIME
        .withColumn(
            "END_TIME",
            F.lit(None).cast("timestamp")
        )

        # Normalize amount unit
        .withColumn(
            "AMOUNT_UNIT",
            F.upper(F.trim(F.col("AMOUNTUOM")))
        )

        # Normalize rate unit
        .withColumn(
            "RATE_UNIT",
            F.upper(F.trim(F.col("RATEUOM")))
        )

        # Normalize route
        .withColumn(
            "ROUTE_NORMALIZED",
            F.upper(F.trim(F.col("ORIGINALROUTE")))
        )

        # Source system
        .withColumn(
            "SOURCE_SYSTEM",
            F.lit("CareVue")
        )
    )

    (
        silver_cv.write
        .mode("overwrite")
        .parquet(SILVER_CV_PATH)
    )

    print_separator("SILVER CV CREATED")

    print(f"Silver CV rows: {silver_cv.count():,}")

    # ------------------------------------------------------------------------
    # CREATE GOLD
    # ------------------------------------------------------------------------

    gold_cv = (

        silver_cv

        .groupBy(
            "SUBJECT_ID",
            "HADM_ID",
            "ITEMID"
        )

        .agg(

            F.first("SOURCE_SYSTEM", True)
                .alias("SOURCE_SYSTEM"),

            F.first("AMOUNT_UNIT", True)
                .alias("AMOUNT_UNIT"),

            F.first("RATE_UNIT", True)
                .alias("RATE_UNIT"),

            F.first("ROUTE_NORMALIZED", True)
                .alias("ROUTE_NORMALIZED"),

            # Event frequency
            F.count("*")
                .alias("EVENT_COUNT"),

            # Amount statistics
            F.avg("AMOUNT")
                .alias("AVG_AMOUNT"),

            F.max("AMOUNT")
                .alias("MAX_AMOUNT"),

            F.min("AMOUNT")
                .alias("MIN_AMOUNT"),

            # Rate statistics
            F.avg("RATE")
                .alias("AVG_RATE"),

            F.max("RATE")
                .alias("MAX_RATE"),

            # Temporal boundaries
            F.min("EVENT_TIME")
                .alias("FIRST_EVENT_TIME"),

            F.max("EVENT_TIME")
                .alias("LAST_EVENT_TIME")
        )
    )

    (
        gold_cv.write
        .mode("overwrite")
        .parquet(GOLD_CV_PATH)
    )

    print_separator("GOLD CV CREATED")

    print(f"Gold CV rows: {gold_cv.count():,}")

    return gold_cv


# ============================================================================
# PART B — INPUTEVENTS_MV
# ============================================================================

def create_mv_pipeline(spark: SparkSession):

    # ------------------------------------------------------------------------
    # LOAD RAW CSV
    # ------------------------------------------------------------------------

    input_mv_raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(INPUT_MV_PATH)
    )

    print_separator("INPUTEVENTS_MV LOADED")

    print(f"Raw MV rows: {input_mv_raw.count():,}")

    input_mv_raw.printSchema()

    input_mv_raw.show(5, truncate=False)

    # ------------------------------------------------------------------------
    # NULL REPORT
    # ------------------------------------------------------------------------

    null_report(input_mv_raw)

    # ------------------------------------------------------------------------
    # CREATE BRONZE
    # ------------------------------------------------------------------------

    (
        input_mv_raw.write
        .mode("overwrite")
        .parquet(BRONZE_MV_PATH)
    )

    print_separator("BRONZE MV CREATED")

    bronze_mv = spark.read.parquet(BRONZE_MV_PATH)

    # ------------------------------------------------------------------------
    # CREATE SILVER
    # ------------------------------------------------------------------------

    silver_mv = (

        bronze_mv

        .dropDuplicates()

        .filter(
            F.col("SUBJECT_ID").isNotNull() &
            F.col("HADM_ID").isNotNull() &
            F.col("ITEMID").isNotNull()
        )

        # Harmonize temporal schema
        .withColumnRenamed(
            "STARTTIME",
            "EVENT_TIME"
        )

        .withColumnRenamed(
            "ENDTIME",
            "END_TIME"
        )

        # Normalize units
        .withColumn(
            "AMOUNT_UNIT",
            F.upper(F.trim(F.col("AMOUNTUOM")))
        )

        .withColumn(
            "RATE_UNIT",
            F.upper(F.trim(F.col("RATEUOM")))
        )

        # Source system
        .withColumn(
            "SOURCE_SYSTEM",
            F.lit("MetaVision")
        )
    )

    (
        silver_mv.write
        .mode("overwrite")
        .parquet(SILVER_MV_PATH)
    )

    print_separator("SILVER MV CREATED")

    print(f"Silver MV rows: {silver_mv.count():,}")

    # ------------------------------------------------------------------------
    # CREATE GOLD
    # ------------------------------------------------------------------------

    gold_mv = (

        silver_mv

        .groupBy(
            "SUBJECT_ID",
            "HADM_ID",
            "ITEMID"
        )

        .agg(

            F.first("SOURCE_SYSTEM", True)
                .alias("SOURCE_SYSTEM"),

            F.first("AMOUNT_UNIT", True)
                .alias("AMOUNT_UNIT"),

            F.first("RATE_UNIT", True)
                .alias("RATE_UNIT"),

            F.first("ORDERCATEGORYNAME", True)
                .alias("ORDER_CATEGORY"),

            # Event frequency
            F.count("*")
                .alias("EVENT_COUNT"),

            # Amount statistics
            F.avg("AMOUNT")
                .alias("AVG_AMOUNT"),

            F.max("AMOUNT")
                .alias("MAX_AMOUNT"),

            F.min("AMOUNT")
                .alias("MIN_AMOUNT"),

            # Rate statistics
            F.avg("RATE")
                .alias("AVG_RATE"),

            F.max("RATE")
                .alias("MAX_RATE"),

            # Temporal boundaries
            F.min("EVENT_TIME")
                .alias("FIRST_EVENT_TIME"),

            F.max("END_TIME")
                .alias("LAST_EVENT_TIME")
        )
    )

    (
        gold_mv.write
        .mode("overwrite")
        .parquet(GOLD_MV_PATH)
    )

    print_separator("GOLD MV CREATED")

    print(f"Gold MV rows: {gold_mv.count():,}")

    return gold_mv


# ============================================================================
# PART C — FUSION
# ============================================================================

def create_fusion_gold(spark: SparkSession):

    gold_cv = spark.read.parquet(GOLD_CV_PATH)

    gold_mv = spark.read.parquet(GOLD_MV_PATH)

    fusion_gold = (

        gold_cv.unionByName(
            gold_mv,
            allowMissingColumns=True
        )
    )

    (
        fusion_gold.write
        .mode("overwrite")
        .parquet(FUSION_GOLD_PATH)
    )

    print_separator("UNIFIED GOLD INPUTEVENTS CREATED")

    print(f"Unified Gold rows: {fusion_gold.count():,}")

    fusion_gold.printSchema()

    fusion_gold.show(10, truncate=False)

    return fusion_gold


# ============================================================================
# FINAL VALIDATION
# ============================================================================

def final_counts():

    bronze_cv = spark.read.parquet(BRONZE_CV_PATH)
    silver_cv = spark.read.parquet(SILVER_CV_PATH)
    gold_cv   = spark.read.parquet(GOLD_CV_PATH)

    bronze_mv = spark.read.parquet(BRONZE_MV_PATH)
    silver_mv = spark.read.parquet(SILVER_MV_PATH)
    gold_mv   = spark.read.parquet(GOLD_MV_PATH)

    fusion_gold = spark.read.parquet(FUSION_GOLD_PATH)

    print_separator("FINAL PIPELINE COUNTS")

    print(f"Bronze CV rows: {bronze_cv.count():,}")
    print(f"Silver CV rows: {silver_cv.count():,}")
    print(f"Gold CV rows:   {gold_cv.count():,}")

    print("-" * 60)

    print(f"Bronze MV rows: {bronze_mv.count():,}")
    print(f"Silver MV rows: {silver_mv.count():,}")
    print(f"Gold MV rows:   {gold_mv.count():,}")

    print("-" * 60)

    print(f"Fusion Gold rows: {fusion_gold.count():,}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print_separator("STARTING INPUTEVENTS PIPELINE")

    # ------------------------------------------------------------------------
    # PART A — CV
    # ------------------------------------------------------------------------

    create_cv_pipeline(spark)

    # ------------------------------------------------------------------------
    # PART B — MV
    # ------------------------------------------------------------------------

    create_mv_pipeline(spark)

    # ------------------------------------------------------------------------
    # PART C — FUSION
    # ------------------------------------------------------------------------

    create_fusion_gold(spark)

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    final_counts()

    print_separator("PIPELINE COMPLETED SUCCESSFULLY")
```
