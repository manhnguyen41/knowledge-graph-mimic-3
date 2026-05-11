import argparse
import os
from typing import Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse selected MIMIC-style clinical event tables."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing CHARTEVENTS.csv.gz, LABEVENTS.csv.gz, MICROBIOLOGYEVENTS.csv.gz, D_ITEMS.csv.gz, D_LABITEMS.csv.gz",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where processed Parquet outputs will be saved.",
    )

    parser.add_argument(
        "--spark-temp-dir",
        type=str,
        default="/tmp/spark",
        help="Directory for Spark temporary files.",
    )

    parser.add_argument(
        "--master",
        type=str,
        default="local[*]",
        help="Spark master. Default: local[*]",
    )

    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Optional row limit for quick testing. Use 0 for full data.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Spark setup
# ---------------------------------------------------------------------

def build_spark(master: str, spark_temp_dir: str) -> SparkSession:
    os.makedirs(spark_temp_dir, exist_ok=True)
    warehouse_dir = os.path.join(os.path.dirname(spark_temp_dir), "warehouse")
    os.makedirs(warehouse_dir, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("MIMICSubsetDataFusion")
        .master(master)
        .config("spark.local.dir", spark_temp_dir)
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def read_csv_gz(spark: SparkSession, path: str, sample_limit: int = 0) -> DataFrame:
    require_file(path)

    df = (
        spark.read
        .option("header", True)
        .option("multiLine", False)
        .option("escape", '"')
        .option("quote", '"')
        .csv(path)
    )

    if sample_limit and sample_limit > 0:
        df = df.limit(sample_limit)

    return df


def clean_text_col(column):
    """
    Spark expression for lightweight text normalization.
    """
    return F.lower(
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(column, r"\[[^\]]*\]", ""),
                    r"\([^)]*\)",
                    "",
                ),
                r"\s+",
                " ",
            )
        )
    )


def safe_double(column):
    """
    Cast to double after trimming. Non-numeric values become NULL.
    """
    return F.trim(column).cast("double")


def write_parquet(df: DataFrame, output_dir: str, name: str):
    path = os.path.join(output_dir, name)
    print(f"[WRITE] {name} -> {path}")
    (
        df.write
        .mode("overwrite")
        .parquet(path)
    )


def print_count(df: DataFrame, name: str):
    try:
        print(f"[COUNT] {name}: {df.count():,}")
    except Exception as e:
        print(f"[WARN] Could not count {name}: {e}")


# ---------------------------------------------------------------------
# Step 1: Join CHARTEVENTS with D_ITEMS
# ---------------------------------------------------------------------

def build_chart_events_named(chartevents: DataFrame, d_items: DataFrame) -> DataFrame:
    """
    Join CHARTEVENTS with D_ITEMS to add human-readable chart item metadata.
    """

    d_items_small = (
        d_items
        .select(
            F.col("ITEMID").alias("d_itemid"),
            F.col("LABEL").alias("chart_label"),
            F.col("ABBREVIATION").alias("chart_abbreviation"),
            F.col("DBSOURCE").alias("dbsource"),
            F.col("LINKSTO").alias("linksto"),
            F.col("CATEGORY").alias("chart_category"),
            F.col("UNITNAME").alias("dictionary_unit"),
            F.col("PARAM_TYPE").alias("param_type"),
            F.col("CONCEPTID").alias("concept_id"),
        )
        .dropDuplicates(["d_itemid"])
    )

    chart_events_named = (
        chartevents.alias("c")
        .join(
            F.broadcast(d_items_small).alias("d"),
            F.col("c.ITEMID") == F.col("d.d_itemid"),
            "left",
        )
        .select(
            F.col("c.ROW_ID").alias("chart_row_id"),
            F.col("c.SUBJECT_ID").alias("subject_id"),
            F.col("c.HADM_ID").alias("hadm_id"),
            F.col("c.ICUSTAY_ID").alias("icustay_id"),
            F.col("c.ITEMID").alias("item_id"),

            F.col("d.chart_label"),
            F.col("d.chart_abbreviation"),
            F.col("d.dbsource"),
            F.col("d.linksto"),
            F.col("d.chart_category"),
            F.col("d.dictionary_unit"),
            F.col("d.param_type"),
            F.col("d.concept_id"),

            F.to_timestamp(F.col("c.CHARTTIME")).alias("chart_time"),
            F.to_timestamp(F.col("c.STORETIME")).alias("store_time"),

            F.col("c.CGID").alias("caregiver_id"),
            F.col("c.VALUE").alias("value_text"),
            safe_double(F.col("c.VALUENUM")).alias("value_num"),
            F.coalesce(F.col("c.VALUEUOM"), F.col("d.dictionary_unit")).alias("value_unit"),

            F.col("c.WARNING").alias("warning"),
            F.col("c.ERROR").alias("error"),
            F.col("c.RESULTSTATUS").alias("result_status"),
            F.col("c.STOPPED").alias("stopped"),
        )
        .withColumn("normalized_label", clean_text_col(F.col("chart_label")))
        .withColumn("normalized_unit", clean_text_col(F.col("value_unit")))
    )

    return chart_events_named


# ---------------------------------------------------------------------
# Step 2: Join LABEVENTS with D_LABITEMS
# ---------------------------------------------------------------------

def build_lab_events_named(labevents: DataFrame, d_labitems: DataFrame) -> DataFrame:
    """
    Join LABEVENTS with D_LABITEMS to add lab item labels, fluid, category, and LOINC code.
    """

    d_labitems_small = (
        d_labitems
        .select(
            F.col("ITEMID").alias("d_itemid"),
            F.col("LABEL").alias("lab_label"),
            F.col("FLUID").alias("fluid"),
            F.col("CATEGORY").alias("lab_category"),
            F.col("LOINC_CODE").alias("loinc_code"),
        )
        .dropDuplicates(["d_itemid"])
    )

    lab_events_named = (
        labevents.alias("l")
        .join(
            F.broadcast(d_labitems_small).alias("d"),
            F.col("l.ITEMID") == F.col("d.d_itemid"),
            "left",
        )
        .select(
            F.col("l.ROW_ID").alias("lab_row_id"),
            F.col("l.SUBJECT_ID").alias("subject_id"),
            F.col("l.HADM_ID").alias("hadm_id"),
            F.col("l.ITEMID").alias("item_id"),

            F.col("d.lab_label"),
            F.col("d.fluid"),
            F.col("d.lab_category"),
            F.col("d.loinc_code"),

            F.to_timestamp(F.col("l.CHARTTIME")).alias("chart_time"),

            F.col("l.VALUE").alias("value_text"),
            safe_double(F.col("l.VALUENUM")).alias("value_num"),
            F.col("l.VALUEUOM").alias("value_unit"),
            F.col("l.FLAG").alias("flag"),
        )
        .withColumn("normalized_label", clean_text_col(F.col("lab_label")))
        .withColumn("normalized_unit", clean_text_col(F.col("value_unit")))
    )

    return lab_events_named


# ---------------------------------------------------------------------
# Step 3: Clean MICROBIOLOGYEVENTS
# ---------------------------------------------------------------------

def build_microbiology_clean(microbio: DataFrame) -> DataFrame:
    """
    Clean MICROBIOLOGYEVENTS.

    This table is not joined to a dictionary in this reduced setup because
    it already contains specimen, organism, antibiotic, and interpretation fields.
    """

    microbiology_clean = (
        microbio
        .select(
            F.col("ROW_ID").alias("micro_row_id"),
            F.col("SUBJECT_ID").alias("subject_id"),
            F.col("HADM_ID").alias("hadm_id"),

            F.to_timestamp(F.col("CHARTDATE")).alias("chart_date"),
            F.to_timestamp(F.col("CHARTTIME")).alias("chart_time"),

            F.col("SPEC_ITEMID").alias("specimen_item_id"),
            F.col("SPEC_TYPE_DESC").alias("specimen_type"),

            F.col("ORG_ITEMID").alias("organism_item_id"),
            F.col("ORG_NAME").alias("organism_name"),

            F.col("ISOLATE_NUM").alias("isolate_num"),

            F.col("AB_ITEMID").alias("antibiotic_item_id"),
            F.col("AB_NAME").alias("antibiotic_name"),

            F.col("DILUTION_TEXT").alias("dilution_text"),
            F.col("DILUTION_COMPARISON").alias("dilution_comparison"),
            safe_double(F.col("DILUTION_VALUE")).alias("dilution_value"),
            F.col("INTERPRETATION").alias("interpretation"),
        )
        .withColumn("event_time", F.coalesce(F.col("chart_time"), F.col("chart_date")))
        .withColumn("normalized_specimen_type", clean_text_col(F.col("specimen_type")))
        .withColumn("normalized_organism_name", clean_text_col(F.col("organism_name")))
        .withColumn("normalized_antibiotic_name", clean_text_col(F.col("antibiotic_name")))
    )

    return microbiology_clean


# ---------------------------------------------------------------------
# Step 4: Fuse chart/lab/microbiology into a unified observation table
# ---------------------------------------------------------------------

def build_clinical_observation(
    chart_events_named: DataFrame,
    lab_events_named: DataFrame,
) -> DataFrame:
    """
    Vertically fuse chart and lab clinical event sources into one common observation table.
    """

    chart_obs = (
        chart_events_named
        .select(
            F.lit("chartevents").alias("source_table"),
            F.col("chart_row_id").alias("source_row_id"),

            F.col("subject_id"),
            F.col("hadm_id"),
            F.col("icustay_id"),

            F.col("chart_time").alias("event_time"),
            F.col("store_time"),

            F.col("item_id"),
            F.lit("chart").alias("item_type"),
            F.col("chart_label").alias("item_label"),
            F.col("normalized_label"),
            F.col("chart_category").alias("category"),

            F.col("value_text"),
            F.col("value_num"),
            F.col("value_unit"),
            F.col("normalized_unit"),

            F.concat_ws(
                "|",
                F.concat(F.lit("warning="), F.coalesce(F.col("warning"), F.lit(""))),
                F.concat(F.lit("error="), F.coalesce(F.col("error"), F.lit(""))),
                F.concat(F.lit("result_status="), F.coalesce(F.col("result_status"), F.lit(""))),
                F.concat(F.lit("stopped="), F.coalesce(F.col("stopped"), F.lit(""))),
            ).alias("event_flag"),
        )
    )

    lab_obs = (
        lab_events_named
        .select(
            F.lit("labevents").alias("source_table"),
            F.col("lab_row_id").alias("source_row_id"),

            F.col("subject_id"),
            F.col("hadm_id"),
            F.lit(None).cast(StringType()).alias("icustay_id"),

            F.col("chart_time").alias("event_time"),
            F.lit(None).cast("timestamp").alias("store_time"),

            F.col("item_id"),
            F.lit("lab").alias("item_type"),
            F.col("lab_label").alias("item_label"),
            F.col("normalized_label"),
            F.col("lab_category").alias("category"),

            F.col("value_text"),
            F.col("value_num"),
            F.col("value_unit"),
            F.col("normalized_unit"),

            F.col("flag").alias("event_flag"),
        )
    )

    clinical_observation = (
        chart_obs
        .unionByName(lab_obs)
        .withColumn(
            "observation_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.col("source_table"),
                    F.col("source_row_id"),
                    F.col("subject_id"),
                    F.coalesce(F.col("hadm_id"), F.lit("NO_HADM")),
                    F.coalesce(F.col("item_id"), F.lit("NO_ITEM")),
                    F.coalesce(F.col("event_time").cast("string"), F.lit("NO_TIME")),
                ),
                256,
            )
        )
        .dropDuplicates(["observation_id"])
    )

    return clinical_observation

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.spark_temp_dir, exist_ok=True)

    spark = build_spark(args.master, args.spark_temp_dir)

    input_files: Dict[str, str] = {
        "chartevents": os.path.join(args.input_dir, "CHARTEVENTS.csv.gz"),
        "labevents": os.path.join(args.input_dir, "LABEVENTS.csv.gz"),
        "microbio": os.path.join(args.input_dir, "MICROBIOLOGYEVENTS.csv.gz"),
        "d_items": os.path.join(args.input_dir, "D_ITEMS.csv.gz"),
        "d_labitems": os.path.join(args.input_dir, "D_LABITEMS.csv.gz"),
    }

    print("[INFO] Reading input files...")
    chartevents = read_csv_gz(spark, input_files["chartevents"], args.sample_limit)
    labevents = read_csv_gz(spark, input_files["labevents"], args.sample_limit)
    microbio = read_csv_gz(spark, input_files["microbio"], args.sample_limit)
    d_items = read_csv_gz(spark, input_files["d_items"], 0)
    d_labitems = read_csv_gz(spark, input_files["d_labitems"], 0)

    print("[INFO] Building joined event tables...")
    chart_events_named = build_chart_events_named(chartevents, d_items)
    lab_events_named = build_lab_events_named(labevents, d_labitems)
    microbiology_clean = build_microbiology_clean(microbio)

    # print("[INFO] Building fused clinical observation table (Chart + Lab)...")
    # clinical_observation = build_clinical_observation(
    #     chart_events_named,
    #     lab_events_named,
    # )

    print("[INFO] Counts:")
    print_count(chart_events_named, "chart_events_named")
    print_count(lab_events_named, "lab_events_named")
    print_count(microbiology_clean, "microbiology_clean")
    # print_count(clinical_observation, "clinical_observation")

    print("[INFO] Writing outputs...")
    write_parquet(chart_events_named, args.output_dir, "chart_events_named")
    write_parquet(lab_events_named, args.output_dir, "lab_events_named")
    write_parquet(microbiology_clean, args.output_dir, "microbiology_clean")
    # write_parquet(clinical_observation, args.output_dir, "clinical_observation")

    print("[DONE] Data fusion completed successfully.")
    spark.stop()


if __name__ == "__main__":
    main()