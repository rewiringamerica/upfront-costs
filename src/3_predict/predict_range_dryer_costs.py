# Databricks notebook source
# MAGIC  %md
# MAGIC
# MAGIC # Upfront Costs Figures for Appliances with "Invariant" costs
# MAGIC
# MAGIC ### Goal
# MAGIC Produce a table of upfront costs for appliances which do not vary by building or location (dryers, ranges), applied to all buildings in a BuildingSet.
# MAGIC
# MAGIC ### Process
# MAGIC 1. Hard-code appliance upfront costs based on desk research. Costs for ER dryers, ER ranges, and induction ranges are from [Home Depot](https://www.homedepot.com/b/Appliances/N-5yc1vZbv1w), [Lowe's](https://www.lowes.com/c/Appliances), and [Best Buy](https://www.bestbuy.com/site/electronics/home-appliances/abcat0900000.c?id=abcat0900000) as of May 2026. These cost estimates do not include installation costs.
# MAGIC 2. Cross-join appliances with buildings
# MAGIC 3. Set upfront costs to 0 for buildings which already have the appliances in question
# MAGIC
# MAGIC ### I/O
# MAGIC ##### Inputs:
# MAGIC - BuildingSet metadata
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `processed.<building_set>.stove_range_dryer_cost_by_building`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

# imports
import pyspark.sql.functions as F
from pyspark.sql.types import *

from dmlutils import sparkdb
from dmlutils.building_upgrades.upgrades import (
    BuildingSet,
    get_building_metadata_table,
    get_table_name_for_building_set,
    national_building_metadata,
)

from src import util

# COMMAND ----------

dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

dbutils.widgets.dropdown(
    "building_set", choices=[b.value for b in BuildingSet], defaultValue=BuildingSet.RESSTOCK_2024_2
)
BUILDING_SET = dbutils.widgets.get("building_set")

# Load current release version so that we can load the proper model number
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md ### Load data

# COMMAND ----------

# Load building metadata
building_metadata_table_path = get_building_metadata_table(BUILDING_SET)
building_set_metadata = spark.table(building_metadata_table_path)

# COMMAND ----------

# MAGIC %md ### Define cost table

# COMMAND ----------

# Define Cost schema
invariant_cost_schema = StructType(
    [
        # What is the appliance?
        StructField("appliance_option", StringType(), False),
        # What are the estimated upfront costs?
        StructField("upfront_cost_median", DoubleType(), False),
        StructField("upfront_cost_20percentile", DoubleType(), True),
        StructField("upfront_cost_80percentile", DoubleType(), True),
    ]
)

range_dryer_costs = spark.createDataFrame(
    [
        ("Electric Resistance Range", 800.0, 600.0, 1200.0),
        ("Electric Resistance Dryer", 650.0, 500.0, 900.0),
        ("Induction Range", 1500.0, 1100.0, 2000.0),
        ("Heat Pump Dryer", 1400.0, 1400.0, 1400.0),
        ("Ventless Heat Pump Dryer", 1400.0, 1400.0, 1400.0),
    ],
    schema=invariant_cost_schema,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join to BuildingSet

# COMMAND ----------

# Define the condition where a building already has the appliance
# NOTE: these are lowest-level conditions that should apply no matter what upgrade eligibility logic is applied downstream.
# These do not replace/represent upgrade eligibility logic.
already_has_appliance_cond = (
    (
        # Don't join induction range costs to buildings with induction ranges already
        (F.col("b.in_electric_cooking_range_type") == "Induction")
        & (F.col("c.appliance_option") == "Induction Range")
    )
    | (
        # Don't join ER range costs to buildings with ER ranges already
        (F.col("b.in_electric_cooking_range_type") == "Electric")
        & (F.col("c.appliance_option") == "Electric Resistance Range")
    )
    | (
        # Don't join ER dryer costs to buildings with an ER dryer already
        (F.col("b.in_clothes_dryer_fuel") == "Electricity")
        & (F.col("c.appliance_option") == "Electric Resistance Dryer")
    )
)

# CrossJoin the tables
range_dryer_cost_by_building = (
    building_set_metadata.alias("b")
    .crossJoin(range_dryer_costs.alias("c"))
    # Conditionally set the costs to $0.0 if the condition is met, otherwise keep the estimated cost
    .withColumn(
        "upfront_cost_median", F.when(already_has_appliance_cond, 0.0).otherwise(F.col("c.upfront_cost_median"))
    )
    .withColumn(
        "upfront_cost_20percentile",
        F.when(already_has_appliance_cond, 0.0).otherwise(F.col("c.upfront_cost_20percentile")),
    )
    .withColumn(
        "upfront_cost_80percentile",
        F.when(already_has_appliance_cond, 0.0).otherwise(F.col("c.upfront_cost_80percentile")),
    )
    .select(
        "building_id",
        "appliance_option",
        "upfront_cost_median",
        "upfront_cost_20percentile",
        "upfront_cost_80percentile",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write out to table

# COMMAND ----------

# Specify BuildingSet in a table column and use selective overwrite for national BuildingSets.
range_dryer_cost_by_building = range_dryer_cost_by_building.withColumn("building_set", F.lit(BUILDING_SET))
write_options = {}
if BUILDING_SET in national_building_metadata:
    write_options = {"overwriteSchema": "false", "replaceWhere": f"building_set = '{BUILDING_SET}'"}

# Determine output table name based on building set
db_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="processed",
    schema="upfront_costs",
    table_name="range_dryer_cost_by_building",
)

# Write out table
sparkdb.write_delta_table(
    df=range_dryer_cost_by_building,
    db_table_name=db_table_name,
    mode=MODE,
    release_version=RELEASE_VERSION,
    **write_options,
)
