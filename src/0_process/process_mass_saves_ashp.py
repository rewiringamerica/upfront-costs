# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Process Mass ASHP Pricing Data
# MAGIC
# MAGIC ### Goal
# MAGIC Process pricing data from Mass CEC 2015-2019 ASHP program (single-zone, partial, and whole-home)
# for use in upfront cost model training.
# MAGIC
# MAGIC ### Process
# MAGIC - Standardize and clean column names
# MAGIC - Extract SEER/HSPF efficiencies from AHRI data and join to Mass heat pump data based on model number
# MAGIC - Create hash id
# MAGIC - Write to delta table
# MAGIC
# MAGIC ### Inputs
# MAGIC - `raw / pricing / mass_ashp_data.csv`
# MAGIC - `raw / pricing / multi_head_efficiency.csv`
# MAGIC - `raw / pricing / single_head_efficiency.csv`
# MAGIC
# MAGIC ### Outputs
# MAGIC - `processed.upfront_costs.mass_ashp`
# MAGIC
# MAGIC ### To Do
# MAGIC - Fix town names for missing records

# COMMAND ----------

# MAGIC %md ### Set Up

# COMMAND ----------

# imports
import pandas as pd
import pyspark.sql.functions as F

# internal
from dmlutils import constants as C
from dmlutils import data_cleaning_utils as dcutil
from dmlutils import hashutils, sparkdb

import src.globals as G
from src import util

# COMMAND ----------

dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

# file paths
PRICING_RAW_DATA_FPATH = G.RAW_DATA_FPATH / "pricing"
PRICING_PROCESSED_DATA_FPATH = G.PROCESSED_DATA_FPATH / "pricing"

# COMMAND ----------

# Load current release version to version feature table written at end of script
RELEASE_VERSION = util.get_release_version()

# get approximate heating tons per ksqft for MA
# this is because MA is in 5A, so 1.7 is the tons per sqft, and then 30% more for a cold (but not dry)
# climate. see upfront_cost_heat_pumps.py for explanation.
ma_hvac_tons_per_ksqft = 1.7 * 1.3

# COMMAND ----------

# MAGIC %md ### Load Data

# COMMAND ----------

# Mass CEC data
mass_ashp_path = PRICING_RAW_DATA_FPATH / "mass_ashp_data.csv"
mass_ashp = pd.read_csv(
    mass_ashp_path,
    converters={
        " Rebate Amount $ ": lambda s: s.replace("$", "").replace(",", ""),
        "  Footage": lambda s: s.replace(",", "")
        .replace("~", "")
        .replace("+", "")
        .replace("'", "")
        .replace("about", ""),
    },
    thousands=",",
    decimal=".",
    dtype="str",
    na_values=[" Not Applicable "],
    nrows=20094,
)

# Nested geographies
nest_geo = spark.table("geographic_crosswalk.nested_geographies_2021")

# COMMAND ----------

# Some preliminary cleaning of column names

# Strip whitespace from column names and rename strange column names
mass_ashp.columns = mass_ashp.columns.str.strip().str.replace("Single- Head", "Single-Head")

# clean column names
mass_ashp = dcutil.clean_colnames(spark.createDataFrame(mass_ashp))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Filling null HSPF and SEER values
# MAGIC We downloaded csvs from AHRI to find SEER/HSPF efficiencies based on the heat pump model number.
# MAGIC After joining based on the model number, there are still null efficiencies. For these, we
# MAGIC take the mean across all SEER/HSPF values and join these.

# COMMAND ----------

# read in the efficiency dataframes from above
single_head_efficiency = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .csv(str(PRICING_RAW_DATA_FPATH / "single_head_efficiency.csv"))
    .select("single_head_heat_pump_number1", "seer", "hspf")
    .withColumnRenamed("seer", "seer_single")
    .withColumnRenamed("hspf", "hspf_single")
)

multi_head_efficiency = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .csv(str(PRICING_RAW_DATA_FPATH / "multi_head_efficiency.csv"))
    .select("multi_head_heat_pump_number1", "seer", "hspf")
    .withColumnRenamed("seer", "seer_multi")
    .withColumnRenamed("hspf", "hspf_multi")
)

# join with mass ashp dataframe on the model name
mass_ashp_efficiencies = mass_ashp.join(single_head_efficiency, "single_head_heat_pump_number1", how="left").join(
    multi_head_efficiency, "multi_head_heat_pump_number1", how="left"
)

# calculate HSPF and SEER means for when there are null values in the joined table
stats = mass_ashp_efficiencies.select(
    F.mean(F.col("hspf_single")).alias("hspf_single_mean"),
    F.mean(F.col("hspf_multi")).alias("hspf_multi_mean"),
    F.mean(F.col("seer_single")).alias("seer_single_mean"),
    F.mean(F.col("seer_multi")).alias("seer_multi_mean"),
).collect()

hspf_mean = (stats[0]["hspf_single_mean"] + stats[0]["hspf_multi_mean"]) / 2
seer_mean = (stats[0]["seer_single_mean"] + stats[0]["seer_multi_mean"]) / 2

# COMMAND ----------

# MAGIC %md ### Clean DataFrame

# COMMAND ----------

# standardize some columns for upfront cost dataframe
cost_mass_ashp = (
    mass_ashp_efficiencies.withColumn("city", F.col("project_town"))
    .withColumnRenamed("project_site_county", "county")
    .withColumnRenamed("received_date", "start_date")
    .withColumn("year", F.concat(F.lit("20"), F.substring(F.col("start_date"), -2, 2)))
    .withColumn("month", F.split(F.col("start_date"), "/").getItem(0))
    .withColumn("month", F.lpad(F.col("month"), 2, "0"))
    .withColumn("square_footage", F.col("footage").cast("double"))
    .withColumn("project_cost_raw", F.col("total_costs").cast("double"))
    .withColumn("state_geoid", F.lit("25"))
    .withColumn("program", F.lit("Mass ASHP"))
    .withColumn("is_cold_climate", F.lit(True))
    .withColumn("product_type", F.lit("Heat Pump HVAC"))
    # assume mostly ductless based on models
    .withColumn("ducted", F.lit(False))
    .withColumn("had_duct_replacement", F.lit(False))
    .withColumn("had_panel_upgrade", F.lit(False))
    # heating capacity in tons is the installed capacity in btu divided by 12000
    .withColumn(
        "heating_capacity_tons",
        F.col("installed_capacity_at_5degreesf").cast("double") / C.BTU_PER_TON,
    )
    # heating capacity > average MA capacity for that square footage is whole-home
    # otherwise <24000 btu is single-zone (24,000 btu is roughly the largest single-zone I saw)
    # all others are partial
    .withColumn(
        "heat_pump_type",
        F.when(
            F.col("installed_capacity_at_5degreesf").cast("double") < 24000,
            F.lit("Single Zone"),
        )
        .when(
            F.col("heating_capacity_tons") > ma_hvac_tons_per_ksqft / 1000 * F.col("square_footage"),
            F.lit("Whole Home"),
        )
        .otherwise(F.lit("Partial")),
    )
    # Use the SEER value for multi head if non missing, otherwise use the single head value, and if
    # both are missing use the average over all non-missing in the dataset
    .withColumn(
        "seer",
        F.when(F.col("seer_multi").isNotNull(), F.col("seer_multi"))
        .when(F.col("seer_single").isNotNull(), F.col("seer_single"))
        .otherwise(seer_mean),
    )
    .withColumn(
        "hspf_region_iv",
        F.when(F.col("hspf_multi").isNotNull(), F.col("hspf_multi"))
        .when(F.col("hspf_single").isNotNull(), F.col("hspf_single"))
        .otherwise(hspf_mean),
    )
    .withColumn("city_state", F.concat(F.initcap(F.col("city")), F.lit(", "), F.lit("MA")))
)

# COMMAND ----------

# Join county names to county geoids
cost_mass_ashp_county_geoid = cost_mass_ashp.withColumn(
    "county_name", F.regexp_replace(F.col("county"), " County", "")
).join(
    (nest_geo.where(F.col("state_abbr") == "MA").select("county_name", "county_geoid").distinct()),
    on="county_name",
)

# COMMAND ----------

# MAGIC %md ### Check and Write Out Data

# COMMAND ----------

# Create a hash ID to uniquely identify projects
cost_mass_ashp_county_geoid = hashutils.add_id_hash_col(
    cost_mass_ashp_county_geoid, cost_mass_ashp_county_geoid.columns, True
)

# COMMAND ----------

# DBTITLE 1,(building id, project, fuel, end use)-level savings table
sparkdb.write_delta_table(
    df=cost_mass_ashp_county_geoid,
    db_table_name="processed.upfront_costs.mass_ashp",
    mode=MODE,
    release_version=RELEASE_VERSION,
)
