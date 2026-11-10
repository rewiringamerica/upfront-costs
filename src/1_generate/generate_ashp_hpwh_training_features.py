# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Upfront Cost HVAC training features
# MAGIC
# MAGIC ### Goal
# MAGIC Union upfront cost data sources to produce a training features table with a common schema
# MAGIC
# MAGIC ### Process
# MAGIC - Union upfront cost data sources based on common columns
# MAGIC - Remove rows with no project cost
# MAGIC - Identify the census region to which each record pertains by joining on state from
# MAGIC the state <> region crosswalk
# MAGIC - Join to producer price index and rsmeans data
# MAGIC - Adjust project costs based on PPI and RSmeans
# MAGIC - Write to delta
# MAGIC
# MAGIC
# MAGIC ### Inputs
# MAGIC - Cleaned TechClean CA upfront costs: `processed.upfront_costs.techclean_ca`
# MAGIC - Cleaned MassSave whole home pilot upfront costs: `processed.upfront_costs.mass_wholehome`
# MAGIC - Cleaned MassSave ashp upfront costs: `processed.upfront_costs.mass_ashp`
# MAGIC - RSMeans price indices: `upfront_cost.rsmeans_county`
# MAGIC - Produce Price Index from Census
# MAGIC
# MAGIC ### Outputs
# MAGIC - `processed.upfront_costs.ashp_hpwh_training_features`
# MAGIC - `processed.upfront_costs.ashp_training_features`

# COMMAND ----------

# MAGIC %md ### Set Up

# COMMAND ----------

# imports
import pyspark.sql.functions as F

from dmlutils import sparkdb

# internal
from dmlutils.datasources.bls import pull_bls_inflation_indices
from dmlutils.qa_utils import assert_unique_primary_key

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

# MAGIC %md ### Define Functions

# COMMAND ----------


def get_most_recent_ppi_rate(ppi_df) -> str:
    """
    Get the Producer Price Index from the most recent month in the dataset.

    Inputs:
    ------
        ppi_df: The PySpark dataframe containing PPI data that has already been preprocessed to add
            month and year columns

    Outputs:
    -------
        str: the most recent PPI as a string
    """
    most_recent_ppi_rate = (ppi_df.orderBy([F.desc("year"), F.desc("month")]).select("producer_price_index")).collect()[
        0
    ]["producer_price_index"]

    return most_recent_ppi_rate


# COMMAND ----------

# MAGIC %md ### Load Data

# COMMAND ----------

# TECHCLEAN CA
ca_tech = spark.table(sparkdb.modify_db_table_name("processed.upfront_costs.techcleanca", MODE))

# MassSave (Whole Home data)
mass_wholehome = spark.table(sparkdb.modify_db_table_name("processed.upfront_costs.mass_wholehome", MODE))

# MassSave (ASHP data)
mass_ashp = spark.table(sparkdb.modify_db_table_name("processed.upfront_costs.mass_ashp", MODE))

# RS Means data
rsmeans = spark.table("upfront_cost.rsmeans_county")

# COMMAND ----------

# MAGIC %md ### Combine into a Single DataFrame

# COMMAND ----------

# columns that all datasets now have in common
common_cols = [
    "city",
    "year",
    "month",
    "state_geoid",
    "county_geoid",
    "heating_capacity_tons",
    "product_type",
    "heat_pump_type",
    "hspf_region_iv",
    "seer",
    "ducted",
    "is_cold_climate",
    "project_cost_raw",
    "id",
]

# select the common columns and do some unions
upfront_cost_data = (
    ca_tech.select(
        *common_cols,
        F.col("living_size_sqft").alias("sqft"),
        "had_panel_upgrade",
        "had_duct_replacement",
        "previous_wh_fuel_type",
    )
    .withColumn("program", F.lit("techclean_ca"))
    .unionByName(
        mass_ashp.select(
            *common_cols,
            F.col("footage").alias("sqft"),
            "had_panel_upgrade",
            "had_duct_replacement",
        ).withColumn("program", F.lit("mass_ashp")),
        allowMissingColumns=True,
    )
    .unionByName(
        mass_wholehome.select(
            *common_cols,
            F.col("square_footage").alias("sqft"),
            "had_panel_upgrade",
            "had_duct_replacement",
        ).withColumn("program", F.lit("mass_wholehome")),
        allowMissingColumns=True,
    )
    # drop any with N/A or 0 for project cost
    .na.drop(subset=["project_cost_raw"])
    .where(F.col("project_cost_raw") > 0)
    # Join to state abbreviations
    .join(
        spark.createDataFrame(G.STATE_GEOIDS[["state_geoid", "state_abbr"]]),
        on="state_geoid",
    )
    # Convert month and year to numeric
    .withColumn("year", F.col("year").cast("double"))
    .withColumn("month", F.col("month").cast("double"))
)

# COMMAND ----------

# MAGIC %md ### Add in Time & Geography Adjustment Factors

# COMMAND ----------

# We will join two additional indices: RS Means
# data (shows comparable correlation with project cost as CPIs), and a producer price index
# corresponding to the HVAC product manufacturing.

# First pull producer price index data
ppi = (
    spark.createDataFrame(pull_bls_inflation_indices(["PCU333415333415"]))
    .withColumn("month", F.col("period").substr(2, 2))
    .withColumn("producer_price_index", F.col("value"))
)

# Join to existing data
upfront_cost_data_rsmeans_ppi = upfront_cost_data.join(
    rsmeans.where((F.col("category") == "fire suppression, plumbing & hvac") & (F.col("component") == "total")).select(
        "county_geoid", "rsmeans_price_index"
    ),
    on="county_geoid",
    how="left",
).join(
    ppi.select("year", "month", "producer_price_index"),
    on=["year", "month"],
    how="left",
)

# COMMAND ----------

# find the most recent PPI rate to scale the prices to current prices
most_recent_ppi = float(get_most_recent_ppi_rate(ppi))

# COMMAND ----------

# MAGIC %md ### Clean and Process DataFrame

# COMMAND ----------

# Convert several features to numeric

# List of columns to be converted to numeric
columns_to_convert = [
    "sqft",
    "producer_price_index",
    "month",
    "year",
    "project_cost_raw",
]

# Convert columns to numeric
for column in columns_to_convert:
    upfront_cost_data_rsmeans_ppi = upfront_cost_data_rsmeans_ppi.withColumn(column, F.col(column).cast("double"))

# COMMAND ----------

# Adjust prices for both time (using PPI) and geography (using RSmeans)
upfront_cost_data_rsmeans_ppi_adjusted = (
    upfront_cost_data_rsmeans_ppi.withColumn("most_recent_ppi", F.lit(most_recent_ppi))
    # Adjust cost by multiplying by ratio of most recent PPI rate to the PPI rate at the time of the project
    # and dividing by RSMeans
    .withColumn(
        "project_cost_adjusted",
        F.col("project_cost_raw")
        * F.col("most_recent_ppi")
        / F.col("producer_price_index")
        / (F.col("rsmeans_price_index") / 100),
    ).filter(F.col("project_cost_adjusted").isNotNull())
)

# COMMAND ----------

# MAGIC %md ### Check & Write Out Data

# COMMAND ----------

# Ensure there are no duplicates
assert_unique_primary_key(upfront_cost_data_rsmeans_ppi_adjusted, ["id"])

# COMMAND ----------

# Write two tables - one for space and water heating and one for space heating only

# Space and water heating
sparkdb.write_delta_table(
    df=upfront_cost_data_rsmeans_ppi_adjusted,
    db_table_name="processed.upfront_costs.ashp_hpwh_training_features",
    mode=MODE,
    release_version=RELEASE_VERSION,
)

# COMMAND ----------

# Space heating only
sparkdb.write_delta_table(
    df=(upfront_cost_data_rsmeans_ppi_adjusted.where(F.col("product_type") == "Heat Pump HVAC")),
    db_table_name="processed.upfront_costs.ashp_training_features",
    mode=MODE,
    release_version=RELEASE_VERSION,
)
