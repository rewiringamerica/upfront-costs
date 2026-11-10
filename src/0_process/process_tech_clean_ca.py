# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Process TECH Clean CA Pricing Data
# MAGIC
# MAGIC ### Goal
# MAGIC Process pricing data from TECH Clean CA (most recent pull was March 19, 2024)
# MAGIC for use in upfront cost model training.
# MAGIC
# MAGIC ### Process
# MAGIC - Clean and standardize column names, replace null values, handle edges cases
# MAGIC - Ensure that no duplicate records are created
# MAGIC - Write to delta table
# MAGIC
# MAGIC ### Inputs
# MAGIC - `raw / pricing / TECHWorkingDataset_Single-Family_2024-03-19.csv`
# MAGIC
# MAGIC ### Outputs
# MAGIC - `processed.upfront_costs.techcleanca`

# COMMAND ----------

# MAGIC %md ### Set Up

# COMMAND ----------

# imports
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# internal
from dmlutils import data_cleaning_utils as dcutil
from dmlutils import hashutils, sparkdb

import src.globals as G
from src import util

# COMMAND ----------

dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

# COMMAND ----------

# Load current release version to version feature table written at end of script
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# paths
PRICING_RAW_DATA_FPATH = G.RAW_DATA_FPATH / "pricing"

# COMMAND ----------

# MAGIC %md ### Load Data

# COMMAND ----------

# read in TECH Clean CA data
options = {"inferSchema": "true", "header": "true"}

techcleanca_raw = (
    spark.read.format("csv")
    .options(**options)
    .load(str(PRICING_RAW_DATA_FPATH / "TECHWorkingDataset_Single-Family_2024-03-19.csv"))
)

techcleanca = dcutil.clean_colnames(techcleanca_raw)

# Nested geographies
nest_geo = spark.table("geographic_crosswalk.nested_geographies_2021")

# COMMAND ----------

# MAGIC %md ### Clean DataFrame

# COMMAND ----------

# standardize some columns for upfront cost training features
column_renaming_dict = {
    "customer_county": "county",
    "customer_city": "city",
    "installation_start_date": "start_date",
    "home_square_footage": "square_footage",
    "total_project_cost_dollars": "project_cost_raw",
    "duct_replacement": "had_duct_replacement",
    "panel_upgrade": "had_panel_upgrade",
}

cost_techcleanca = (
    techcleanca
    # Standardize column names
    .withColumnsRenamed(column_renaming_dict)
    # Fix formatting on start date column
    .withColumn("start_date", F.to_timestamp("start_date", "yyyy-MM-dd HH:mm:ss"))
    # Add some additional columns
    .withColumn("state_geoid", F.lit("06"))
    .withColumn("program", F.lit("TECH Clean CA"))
    .withColumn("is_cold_climate", F.lit(False))
    .withColumn("heat_pump_type", F.lit("Whole Home"))
    # both heat pump & heat pump water heater costs are in this dataframe, need to differentiate
    .withColumn(
        "product_type",
        F.when(
            F.col("program_name") == "TECH Clean California - Heat Pump Water Heating",
            F.lit("Heat Pump Water Heater"),
        ).otherwise(F.lit("Heat Pump HVAC")),
    )
    .withColumn("year", F.year(F.to_date(F.col("start_date"), "yyyy-MM")))
    .withColumn("month", F.col("start_date").substr(6, 2))
    # Handle one edge case where Huntington Beach Orange County is coded as "Huntington County"
    .withColumn(
        "county",
        F.when((F.col("county") == "Huntington County"), "Orange County").otherwise(F.col("county")),
    )
    # Create city_state combo
    .withColumn("city_state", F.concat(F.initcap(F.col("city")), F.lit(", "), F.lit("CA")))
)

# Replace null values
# Define values to replace with null
values_to_replace = ["Removed during QA", "", " ", "Not provided"]

# Replace the specified values with null for string column
# s
cost_techcleanca = cost_techcleanca.replace(
    values_to_replace,
    None,
    subset=[c for c in cost_techcleanca.columns if cost_techcleanca.schema[c].dataType == StringType()],
)

# Join county names to county geoids
cost_techcleanca_county_geoid = cost_techcleanca.withColumn(
    "county_name", F.regexp_replace(F.col("county"), " County", "")
).join(
    (nest_geo.where(F.col("state_abbr") == "CA").select("county_name", "county_geoid").distinct()),
    on="county_name",
)

# COMMAND ----------

# MAGIC %md ### Check and Write Out Data

# COMMAND ----------

# Create a hash ID to uniquely identify projects
cost_techcleanca_geo = hashutils.add_id_hash_col(
    cost_techcleanca_county_geoid, cost_techcleanca_county_geoid.columns, True
)

# COMMAND ----------

# DBTITLE 1,(building id, project, fuel, end use)-level savings table
sparkdb.write_delta_table(
    df=cost_techcleanca_geo,
    db_table_name="processed.upfront_costs.techcleanca",
    mode=MODE,
    release_version=RELEASE_VERSION,
)
