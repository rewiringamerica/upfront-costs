# Databricks notebook source
# MAGIC %md # Combine Upfront Costs for Different Appliances/Upgrades
# MAGIC
# MAGIC ### Goal
# MAGIC Create a table which contains upfront cost predictions for every upgrade/appliance, for every building in the given BuildingSet.
# MAGIC
# MAGIC ### Process
# MAGIC This notebook combines the upfront cost tables listed below into a single table of upfront costs by building, appliance, and upgrade.
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs:
# MAGIC - `processed.<building_set>.ashp_upfront_cost_by_building_appliance_upgrade`: ASHP upfront cost predictions for all occupied single-family and small-multifamily occupied homes in the BuildingSet
# MAGIC - `processed.<building_set>.ev_upfront_cost_by_building_vehicle_category_range`: EV upfront cost predictions (for each ev option and building)
# MAGIC - `processed.<building_set>.gshp_upfront_cost_by_building_appliance_upgrade`: GSHP upfront cost predictions for all buildings in the BuildingSet, if applicable
# MAGIC - `processed.<building_set>.hpwh_wiring_panel_cost_by_building`: upfront cost predictions for HPWH, wiring, and electric panels for all buildings in the BuildingSet
# MAGIC - `processed.<building_set>.solar_upfront_cost_by_building_solarconfig`: Rooftop solar costs for all modeled BuildingSet buildings x solar configs
# MAGIC - `processed.<building_set>.weatherization_cost_by_building_package`: Weatherization costs by building and weatherization package
# MAGIC - `processed.<building_set>.stove_range_dryer_cost_by_building`: Costs by building for appliances whose cost is assumed to not vary by location or any other housing characteristics (stoves, ranges, dryers)
# MAGIC -  BuildingSet metadata: used to get counties represented in the BuildingSet
# MAGIC
# MAGIC Note that all input costs that vary by location have been localized already.
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `prepared.<building_set>.upfront_costs_by_building_appliance_upgrade`: Table of upfront costs by building, appliance, and upgrade. Note that ASHP upfront costs are calculated only for occupied single-family and small multifamily homes.
# MAGIC
# MAGIC ### TODOs:
# MAGIC
# MAGIC #### Future Work
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import packages
from functools import reduce

import pyspark.sql.functions as F

from dmlutils import sparkdb
from dmlutils.building_upgrades.upgrades import (
    BuildingSet,
    get_building_metadata_table,
    get_table_name_for_building_set,
    national_building_metadata,
)
from dmlutils.qa_utils import assert_unique_primary_key, check_for_null_values

from src import util

# Load current release version so that we can load the proper model number
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# DBTITLE 1,Widgets

# Set I/O mode
dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

# Choose BuildingSet
dbutils.widgets.dropdown(
    "building_set", choices=[b.value for b in BuildingSet], defaultValue=BuildingSet.RESSTOCK_2024_2
)
BUILDING_SET = dbutils.widgets.get("building_set")

# COMMAND ----------

# DBTITLE 1,Parameters and percentile intervals
# we are currently using the 20th - 80th percentile
LOWER_BOUND_PERCENTILE_COL = "upfront_cost_20percentile"
UPPER_BOUND_PERCENTILE_COL = "upfront_cost_80percentile"

# COMMAND ----------

# MAGIC %md ## Read in Data

# COMMAND ----------

# DBTITLE 1,Load data
# Define common args for retrieving cost prediction tables
table_def_args = {
    "building_set": BUILDING_SET,
    "catalog": "processed",
    "schema": "upfront_costs",
    "mode": MODE,
}

# Upfront cost prediction tables
ashp_table_path = get_table_name_for_building_set(
    table_name="ashp_upfront_cost_by_building_appliance_upgrade", **table_def_args
)
if spark.catalog.tableExists(ashp_table_path):
    ashp_predicted_costs = spark.table(ashp_table_path).filter(F.col("building_set") == BUILDING_SET)
else:
    print("Warning: ASHP upfront cost predictions not available for this BuildingSet.")
    ashp_predicted_costs = None

gshp_table_path = get_table_name_for_building_set(
    table_name="gshp_upfront_cost_by_building_appliance_upgrade", **table_def_args
)
if spark.catalog.tableExists(gshp_table_path):
    gshp_predicted_costs = spark.table(gshp_table_path).filter(F.col("building_set") == BUILDING_SET)
else:
    print("Warning: GSHP upfront cost predictions not available for this BuildingSet.")
    gshp_predicted_costs = None

hpwh_wiring_panel_charger_predicted_costs = spark.table(
    get_table_name_for_building_set(
        table_name="hpwh_wiring_panel_charger_cost_by_building",
        **table_def_args,
    )
).filter(F.col("building_set") == BUILDING_SET)
solar_predicted_costs = spark.table(
    get_table_name_for_building_set(
        table_name="solar_upfront_cost_by_building_solarconfig",
        **table_def_args,
    )
).filter(F.col("building_set") == BUILDING_SET)
weatherization_predicted_costs = spark.table(
    get_table_name_for_building_set(
        table_name="weatherization_cost_by_building_package",
        **table_def_args,
    )
).filter(F.col("building_set") == BUILDING_SET)
ev_predicted_costs = spark.table(
    get_table_name_for_building_set(
        table_name="ev_upfront_cost_by_building_vehicle_category_range",
        **table_def_args,
    )
).filter(F.col("building_set") == BUILDING_SET)
range_dryer_costs = spark.table(
    get_table_name_for_building_set(
        table_name="range_dryer_cost_by_building",
        **table_def_args,
    )
).filter(F.col("building_set") == BUILDING_SET)

# COMMAND ----------

# BuildingSet metadata
building_metadata_table_path = get_building_metadata_table(BUILDING_SET)
building_set_metadata = spark.table(building_metadata_table_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocess tables

# COMMAND ----------

id_col = "building_id"
attribute_cols = [
    "upgrade_id",
    "appliance_option",
    LOWER_BOUND_PERCENTILE_COL,
    "upfront_cost_median",
    UPPER_BOUND_PERCENTILE_COL,
]

# COMMAND ----------

# DBTITLE 1,Preprocess cost tables
if ashp_predicted_costs:
    ashp_predicted_costs = ashp_predicted_costs.select(id_col, *attribute_cols)

hpwh_wiring_panel_charger_predicted_costs = hpwh_wiring_panel_charger_predicted_costs.withColumn(
    "upgrade_id",
    # For now, HPWH, panel upgrade and wiring costs aren't associated with any specific upgrade
    F.lit(None).cast("double"),
).select(id_col, *attribute_cols)

solar_predicted_costs = (
    solar_predicted_costs.withColumn("appliance_option", F.lit("Rooftop Solar"))
    # For now, solar costs aren't associated with any specific upgrade
    .withColumn("upgrade_id", F.lit(None).cast("double")).select(id_col, "solar_upgrade_id", *attribute_cols)
    # Some buildings have Null solar cost predictions, generally due to not having roofs in ResStock. Drop these.
    .dropna(subset=[LOWER_BOUND_PERCENTILE_COL, "upfront_cost_median", UPPER_BOUND_PERCENTILE_COL])
)

if gshp_predicted_costs:
    gshp_predicted_costs = (
        gshp_predicted_costs.withColumn(LOWER_BOUND_PERCENTILE_COL, F.lit(None).cast("double"))
        .withColumn(UPPER_BOUND_PERCENTILE_COL, F.lit(None).cast("double"))
        .select(id_col, *attribute_cols)
    )

weatherization_predicted_costs = (
    weatherization_predicted_costs.withColumnsRenamed({"weatherization_package": "appliance_option"})
    # Clarify appliance option name for weatherization measures
    .withColumn("appliance_option", F.concat(F.col("appliance_option"), F.lit(" Weatherization")))
    # Weatherization alone isn't associated with any specific upgrade
    .withColumn("upgrade_id", F.lit(None).cast("double"))
    # For now we don't estimate upfront cost variability for weatherization
    .withColumn(LOWER_BOUND_PERCENTILE_COL, F.col("upfront_cost_median"))
    .withColumn(UPPER_BOUND_PERCENTILE_COL, F.col("upfront_cost_median"))
    .select(id_col, *attribute_cols)
)

ev_predicted_costs = ev_predicted_costs.withColumnsRenamed({"in_electric_vehicle_battery": "appliance_option"}).select(
    id_col, *attribute_cols
)

range_dryer_costs = range_dryer_costs.withColumn(
    # For now, range and dryer costs aren't associated with any specific upgrade
    "upgrade_id",
    F.lit(None).cast("double"),
).select(id_col, *attribute_cols)

# COMMAND ----------

# MAGIC %md ## Combine upfront costs

# COMMAND ----------

# Combine varying costs
cost_tables = [
    table
    for table in [
        ashp_predicted_costs,
        gshp_predicted_costs,
        hpwh_wiring_panel_charger_predicted_costs,
        weatherization_predicted_costs,
        ev_predicted_costs,
        range_dryer_costs,
    ]
    if table is not None
]
upfront_cost_all = reduce(
    lambda left, right: left.unionByName(right),
    cost_tables,
)
# Add in solar costs, which include `solar_upgrade_id` column
upfront_cost_all = upfront_cost_all.unionByName(solar_predicted_costs, allowMissingColumns=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check data assumptions

# COMMAND ----------

# No duplicate building_id, upgrade_id, appliance_option, solar_upgrade_id combos
assert_unique_primary_key(upfront_cost_all, [id_col, "upgrade_id", "appliance_option", "solar_upgrade_id"])
# No null values
assert not (
    check_for_null_values(upfront_cost_all.select(id_col, "appliance_option", "upfront_cost_median"), count_nulls=False)
), "Unexpected Null values"

# No negative costs
assert upfront_cost_all.filter(
    (F.col("upfront_cost_20percentile") < 0)
    | (F.col("upfront_cost_median") < 0)
    | (F.col("upfront_cost_80percentile") < 0)
).isEmpty(), "Negative costs"

# Percentiles are monotonic
assert upfront_cost_all.filter(
    (F.col("upfront_cost_20percentile") > F.col("upfront_cost_median"))
    | (F.col("upfront_cost_median") > F.col("upfront_cost_80percentile"))
).isEmpty(), "Non-monotonic percentiles"

# COMMAND ----------

# MAGIC %md ## Write out to Delta

# COMMAND ----------

# DBTITLE 1,Write out to Delta tables
write_options = {}
if BUILDING_SET in national_building_metadata:
    # If using a national building set, specify in a table column and use selective overwrite
    upfront_cost_all = upfront_cost_all.withColumn("building_set", F.lit(BUILDING_SET))
    write_options = {"overwriteSchema": "false", "replaceWhere": f"building_set = '{BUILDING_SET}'"}

# Determine output table name based on building set
db_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="prepared",
    schema="upfront_costs",
    table_name="upfront_costs_by_building_appliance_upgrade",
)

# Write out table
sparkdb.write_delta_table(
    df=upfront_cost_all,
    db_table_name=db_table_name,
    mode=MODE,
    release_version=RELEASE_VERSION,
    **write_options,
)
