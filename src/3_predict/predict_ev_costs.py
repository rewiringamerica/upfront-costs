# Databricks notebook source
# MAGIC  %md
# MAGIC
# MAGIC # Upfront Costs Figures for EVs by Building
# MAGIC
# MAGIC ### Goal
# MAGIC Produce a table of EV upfront costs by building for a given BuildingSet.
# MAGIC
# MAGIC ### Process
# MAGIC - Load processed EV data from process_ev_data.py, which contains EV upfront costs by vehicle category and range.
# MAGIC - Join to building metadata to produce a table of EV upfront costs by building.
# MAGIC - Postprocessing: for newer BuildingSets, set estimated costs to 0 for the buildings with that EV already.
# MAGIC
# MAGIC ### I/O
# MAGIC ##### Inputs:
# MAGIC - BuildingSet metadata
# MAGIC - `processed.<building_set>.ev_upfront_cost_by_vehicle_category_range`
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `processed.<building_set>.ev_upfront_cost_by_building_vehicle_category_range`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

# imports
import pyspark.sql.functions as F

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

# Processed EV costs
ev_upfront_cost_by_vehicle_category_range = spark.table(
    sparkdb.modify_db_table_name(
        db_table_name="processed.upfront_costs.ev_upfront_cost_by_vehicle_category_range",
        mode=MODE,
    )
)

# COMMAND ----------

# MAGIC %md ### Process EV upfront costs data

# COMMAND ----------

# For now we don't associate EVs with any specific upgrade
ev_upfront_cost_by_vehicle_category_range = ev_upfront_cost_by_vehicle_category_range.withColumn(
    "upgrade_id", F.lit(None).cast("double")
)
# Create `in_electric_vehicle_battery` column for joining to building metadata
ev_upfront_cost_by_vehicle_category_range = ev_upfront_cost_by_vehicle_category_range.withColumn(
    "in_electric_vehicle_battery",
    F.concat(F.col("vehicle_category"), F.lit(", Battery Electric Vehicle, "), F.col("range"), F.lit(" mile range")),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join to BuildingSet

# COMMAND ----------

# We want to produce estimated costs of all EV types for all buildings,
# agnostic to which EV type ResStock estimates would be purchased for that building.
if BUILDING_SET in [BuildingSet.RESSTOCK_2024_2, BuildingSet.RESSTOCK_2022_1, BuildingSet.SAN_JOSE_CA_2024_2]:
    ev_upfront_cost_by_building_vehicle_category_range = ev_upfront_cost_by_vehicle_category_range.join(
        building_set_metadata.select("building_id"),
        how="cross",
    )
else:
    # For newer BuildingSets, set estimated costs to 0 for the buildings with that EV already
    ev_upfront_cost_by_building_vehicle_category_range = ev_upfront_cost_by_vehicle_category_range.alias("e").join(
        building_set_metadata.select(
            "building_id", "in_electric_vehicle_battery", "in_electric_vehicle_ownership"
        ).alias("b"),
        how="cross",
    )
    for column in ["upfront_cost_median", "upfront_cost_20percentile", "upfront_cost_80percentile"]:
        ev_upfront_cost_by_building_vehicle_category_range = (
            ev_upfront_cost_by_building_vehicle_category_range.withColumn(
                column,
                F.when(
                    (F.col("e.in_electric_vehicle_battery") == F.col("b.in_electric_vehicle_battery"))
                    & (F.col("b.in_electric_vehicle_ownership") == "Yes"),
                    0.0,
                ).otherwise(F.col(column)),
            )
        )
    # Drop the original EV column so the final table lists the upgrade EV
    ev_upfront_cost_by_building_vehicle_category_range = ev_upfront_cost_by_building_vehicle_category_range.drop(
        F.col("b.in_electric_vehicle_battery"), F.col("b.in_electric_vehicle_ownership")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write out to table

# COMMAND ----------

# Specify BuildingSet in a table column and use selective overwrite for national BuildingSets.
ev_upfront_cost_by_building_vehicle_category_range = ev_upfront_cost_by_building_vehicle_category_range.withColumn(
    "building_set", F.lit(BUILDING_SET)
)
write_options = {}
if BUILDING_SET in national_building_metadata:
    write_options = {"overwriteSchema": "false", "replaceWhere": f"building_set = '{BUILDING_SET}'"}

# Determine output table name based on building set
db_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="processed",
    schema="upfront_costs",
    table_name="ev_upfront_cost_by_building_vehicle_category_range",
)

# Write out table
sparkdb.write_delta_table(
    df=ev_upfront_cost_by_building_vehicle_category_range,
    db_table_name=db_table_name,
    mode=MODE,
    release_version=RELEASE_VERSION,
    **write_options,
)
