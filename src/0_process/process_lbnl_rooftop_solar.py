# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Ingest Lawrence Berkeley National Lab Tracking the Sun Data
# MAGIC
# MAGIC ### Goal
# MAGIC Process installed rooftop solar costs from [LBNL Tracking the Sun Report](https://eta-publications.lbl.gov/sites/default/files/2024-10/tracking_the_sun_2024_report.pdf),
# MAGIC a dataset on the characteristics of roughly 75% of all distributed PV installations
# MAGIC in the US since 2002.
# MAGIC
# MAGIC ### Process
# MAGIC Requests the zipped file from a static URL. Unzips the file within Google Cloud and
# MAGIC saves the CSV. Reads in raw CSV. Formats relevant columns in unified format.
# MAGIC Saves as processed delta table.
# MAGIC
# MAGIC ### Outputs
# MAGIC - `raw / lbnl / tracking_the_sun / TTS_LBNL_public_file_29-Sep-2025_all.csv`
# MAGIC - `processed.upfront_costs.lbnl_rooftop_solar`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup

# COMMAND ----------

# imports
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from dmlutils import sparkdb

# internal
import src.globals as G
from src import util

# COMMAND ----------

dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

# Load current release version to version feature table written at end of script
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the data

# COMMAND ----------

# Define data path
data_path = G.RAW_DATA_FPATH / "lbnl" / "tracking_the_sun" / "TTS_LBNL_public_file_29-Sep-2025_all.csv"

# Load rooftop solar data from raw data file in the cloud
lbnl_solar_df = spark.read.csv(str(data_path), header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean and process the data

# COMMAND ----------

# Filter down to columns of interest
rooftop_solar_df = lbnl_solar_df.select(
    [
        "installation_date",
        col("PV_system_size_DC").alias("system_size_kw"),
        "total_installed_price",
        "customer_segment",
        "zip_code",
        col("state").alias("state_abbr"),
        "tracking",
        "ground_mounted",
        "self_installed",
        "third_party_owned",
        "rebate_or_grant",
    ]
)

# COMMAND ----------

# Add in date columns and cost per watt installed columns
rooftop_solar_df = (
    rooftop_solar_df.withColumn("date", F.to_date(col("installation_date"), "yyyy-MM-dd"))
    .withColumn("month", F.month(col("date").alias("month")))
    .withColumn("year", F.year(col("date").alias("year")))
    .withColumn("cost_dollars_per_watt_installed", col("total_installed_price") / (col("system_size_kw") * 1000))
    .drop("date")
)

# COMMAND ----------

# Filter out entries that are not for helpful for residential upfront cost calculations
# Data dictionary can be found as the .docx file found in this download
# https://emp.lbl.gov/sites/default/files/public_datafile.zip
rooftop_solar_df = rooftop_solar_df.filter(
    # we want residential installations and not commercial or government
    (col("customer_segment") == "RES")
    &
    # we do not want systems that track the sun, this is a capability most roof mounted solar lacks
    (col("tracking") != 1)
    &
    # we only want roof mounted systems, not ground mounted systems
    (col("ground_mounted") != 1)
    &
    # residents might self install but it's difficult and rare so we don't want to affect the
    # predictions since most will be done by a contractor
    (col("self_installed") != 1)
    &
    # we need system size and location information for this to work
    (col("zip_code") != -1)
    & (col("system_size_kw") != -1)
    # only installs with known, positive nonzero costs are considered
    & (col("total_installed_price") > 0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to GCFS

# COMMAND ----------

sparkdb.write_delta_table(
    df=rooftop_solar_df,
    db_table_name="processed.upfront_costs.lbnl_rooftop_solar",
    mode=MODE,
    release_version=RELEASE_VERSION,
)
