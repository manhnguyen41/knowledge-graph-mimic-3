```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

# ============================================================================
# CONFIG
# ============================================================================

INPUT_PATH = "/tmp/PRESCRIPTIONS.csv.gz"

BRONZE_PATH = "/tmp/bronze_prescriptions"
SILVER_PATH = "/tmp/silver_prescriptions"
GOLD_PATH   = "/tmp/gold_prescriptions"

# ============================================================================
# SPARK SESSION
# ============================================================================

spark = (
    SparkSession.builder
    .appName("MIMICIII_PRESCRIPTIONS_PIPELINE")
    .getOrCreate()
)

# ============================================================================
# STEP 1 — LOAD RAW CSV
# ============================================================================

def load_raw_data(spark: SparkSession) -> DataFrame:

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(INPUT_PATH)
    )

    print("=" * 60)
    print("RAW PRESCRIPTIONS LOADED")
    print("=" * 60)

    print(f"Raw rows: {df.count():,}")

    return df


# ============================================================================
# STEP 2 — CREATE BRONZE
# ============================================================================

def create_bronze(spark: SparkSession) -> DataFrame:

    df_raw = load_raw_data(spark)

    (
        df_raw.write
        .mode("overwrite")
        .parquet(BRONZE_PATH)
    )

    print("=" * 60)
    print("BRONZE CREATED")
    print("=" * 60)

    return spark.read.parquet(BRONZE_PATH)


# ============================================================================
# STEP 3 — BRONZE DATA INSPECTION
# ============================================================================

def inspect_bronze(df: DataFrame) -> None:

    print("=" * 60)
    print("BRONZE SCHEMA")
    print("=" * 60)

    df.printSchema()

    print("=" * 60)
    print("BRONZE SAMPLE")
    print("=" * 60)

    df.show(5, truncate=False)

    print("=" * 60)
    print("BRONZE ROW COUNT")
    print("=" * 60)

    print(df.count())


# ============================================================================
# STEP 4 — NULL REPORT
# ============================================================================

def null_report(df: DataFrame) -> None:

    total_rows = df.count()

    print("=" * 60)
    print("NULL REPORT")
    print("=" * 60)

    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])

    null_counts.show(truncate=False)

    print(f"Total rows: {total_rows:,}")


# ============================================================================
# STEP 5 — CREATE SILVER
# ============================================================================

def create_silver(spark: SparkSession) -> DataFrame:

    bronze = spark.read.parquet(BRONZE_PATH)

    silver = (

        bronze

        # Remove exact duplicates
        .dropDuplicates()

        # Remove invalid rows
        .filter(
            F.col("SUBJECT_ID").isNotNull() &
            F.col("HADM_ID").isNotNull() &
            F.col("DRUG").isNotNull()
        )

        # Canonical medication name
        .withColumn(
            "MEDICATION_NAME",
            F.upper(
                F.trim(
                    F.coalesce(
                        F.col("DRUG_NAME_GENERIC"),
                        F.col("DRUG_NAME_POE"),
                        F.col("DRUG")
                    )
                )
            )
        )

        # Canonical medication code
        .withColumn(
            "MEDICATION_CODE",
            F.coalesce(
                F.col("NDC").cast("string"),
                F.col("GSN"),
                F.col("FORMULARY_DRUG_CD")
            )
        )

        # Normalize route
        .withColumn(
            "ADMIN_ROUTE",
            F.upper(F.trim(F.col("ROUTE")))
        )

        # Parse numeric dose
        .withColumn(
            "DOSE_NUMERIC",
            F.regexp_extract(
                F.col("DOSE_VAL_RX"),
                r"(\\d+\\.?\\d*)",
                1
            ).cast("double")
        )

        # Prescription duration
        .withColumn(
            "DURATION_HOURS",
            (
                F.unix_timestamp(F.col("ENDDATE")) -
                F.unix_timestamp(F.col("STARTDATE"))
            ) / 3600
        )
    )

    (
        silver.write
        .mode("overwrite")
        .parquet(SILVER_PATH)
    )

    print("=" * 60)
    print("SILVER CREATED")
    print("=" * 60)

    print(f"Silver rows: {silver.count():,}")

    print(
        f"Distinct medications: "
        f"{silver.select('MEDICATION_NAME').distinct().count():,}"
    )

    return spark.read.parquet(SILVER_PATH)


# ============================================================================
# STEP 6 — CREATE GOLD
# ============================================================================

def create_gold(spark: SparkSession) -> DataFrame:

    silver_df = spark.read.parquet(SILVER_PATH)

    gold = (

        silver_df

        .groupBy(

            # Patient identity
            "SUBJECT_ID",

            # Admission identity
            "HADM_ID",

            # Canonical medication
            "MEDICATION_NAME"
        )

        .agg(

            # Representative medication code
            F.first("MEDICATION_CODE", True)
                .alias("MEDICATION_CODE"),

            # Medication metadata
            F.first("DRUG_TYPE", True)
                .alias("DRUG_TYPE"),

            F.first("ADMIN_ROUTE", True)
                .alias("ADMIN_ROUTE"),

            F.first("DOSE_UNIT_RX", True)
                .alias("DOSE_UNIT_RX"),

            F.first("FORM_UNIT_DISP", True)
                .alias("FORM_UNIT_DISP"),

            F.first("PROD_STRENGTH", True)
                .alias("PROD_STRENGTH"),

            # Frequency
            F.count("*")
                .alias("PRESCRIPTION_COUNT"),

            # ICU coverage
            F.countDistinct("ICUSTAY_ID")
                .alias("ICU_STAY_COUNT"),

            # Dose statistics
            F.avg("DOSE_NUMERIC")
                .alias("AVG_DOSE"),

            F.max("DOSE_NUMERIC")
                .alias("MAX_DOSE"),

            F.min("DOSE_NUMERIC")
                .alias("MIN_DOSE"),

            # Duration statistics
            F.avg("DURATION_HOURS")
                .alias("AVG_DURATION"),

            F.max("DURATION_HOURS")
                .alias("MAX_DURATION"),

            # Temporal boundaries
            F.min("STARTDATE")
                .alias("FIRST_PRESCRIPTION_TIME"),

            F.max("ENDDATE")
                .alias("LAST_PRESCRIPTION_TIME")
        )
    )

    (
        gold.write
        .mode("overwrite")
        .parquet(GOLD_PATH)
    )

    print("=" * 60)
    print("GOLD PRESCRIPTIONS CREATED")
    print("=" * 60)

    print(f"Gold rows: {gold.count():,}")

    gold.show(10, truncate=False)

    return spark.read.parquet(GOLD_PATH)


# ============================================================================
# STEP 7 — VALIDATE GOLD
# ============================================================================

def validate_gold(spark: SparkSession) -> None:

    silver_df = spark.read.parquet(SILVER_PATH)
    gold_df   = spark.read.parquet(GOLD_PATH)

    silver_count = silver_df.count()

    gold_prescription_sum = (
        gold_df
        .agg(
            F.sum("PRESCRIPTION_COUNT")
            .alias("TOTAL_PRESCRIPTION_COUNT")
        )
        .collect()[0][0]
    )

    print("=" * 60)
    print("GOLD VALIDATION")
    print("=" * 60)

    print(f"Silver rows: {silver_count:,}")

    print(
        f"Sum of PRESCRIPTION_COUNT in Gold: "
        f"{gold_prescription_sum:,}"
    )

    print(
        f"Difference: "
        f"{silver_count - gold_prescription_sum:,}"
    )

    print("=" * 60)


# ============================================================================
# STEP 8 — FINAL COUNTS
# ============================================================================

def final_counts(spark: SparkSession) -> None:

    bronze = spark.read.parquet(BRONZE_PATH)
    silver = spark.read.parquet(SILVER_PATH)
    gold   = spark.read.parquet(GOLD_PATH)

    print("=" * 60)
    print("PIPELINE COUNTS")
    print("=" * 60)

    print(f"Bronze count: {bronze.count():,}")
    print(f"Silver count: {silver.count():,}")
    print(f"Gold count:   {gold.count():,}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Bronze
    bronze_df = create_bronze(spark)

    # Inspect Bronze
    inspect_bronze(bronze_df)

    # Null report
    null_report(bronze_df)

    # Silver
    silver_df = create_silver(spark)

    # Gold
    gold_df = create_gold(spark)

    # Validate
    validate_gold(spark)

    # Final counts
    final_counts(spark)

    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
```
