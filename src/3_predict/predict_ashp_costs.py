# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Predict ASHP Upfront Costs on the given BuildingSet
# MAGIC
# MAGIC ### Goal
# MAGIC Run the trained Upfront Cost model on the data for each occupied single-family and small-multifamily building in the given BuildingSet.
# MAGIC
# MAGIC ### I/O
# MAGIC ##### Inputs:
# MAGIC - BuildingSet metadata
# MAGIC - `geographic_crosswalk.resstock_county_to_other_geographies`
# MAGIC - `processed.<building_set>.resstock_ra_outputs_annual`
# MAGIC - `upfront_cost.rsmeans_county`
# MAGIC - Pretrained Quantile Regressor Model tagged with current release version number
# MAGIC ##### Parameters:
# MAGIC - `mode`: mode to write delta tables in (see `sparkdb.modify_db_table_name()`)
# MAGIC - `building_set`: Which BuildingSet to predict costs for
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `processed.<building_set>.ashp_upfront_cost_by_building_appliance_upgrade`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Python imports

# COMMAND ----------

# DBTITLE 0,Import packages
from itertools import chain

import mlflow
import pandas as pd
import pyspark.sql.functions as F
from mlflow.tracking import MlflowClient

from dmlutils import constants as C
from dmlutils import sparkdb
from dmlutils.building_upgrades.heat_pumps import HEAT_PUMPS_BY_NAME
from dmlutils.building_upgrades.upgrades import (
    REGIONAL_UPGRADES,
    UPGRADES,
    BuildingSet,
    get_building_metadata_table,
    get_table_name_for_building_set,
    national_building_metadata,
)

from src import util

# COMMAND ----------

# DBTITLE 1,Widgets
# I/O mode
dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")
# BuildingSet
dbutils.widgets.dropdown(
    "building_set", choices=[b.value for b in BuildingSet], defaultValue=BuildingSet.RESSTOCK_2024_2
)
BUILDING_SET = dbutils.widgets.get("building_set")

# If using a regional BuildingSet, use the list of regional upgrades; otherwise, use the full list of upgrades
ALL_UPGRADES = REGIONAL_UPGRADES.get(BUILDING_SET, UPGRADES)
ALL_UPGRADES_BY_ID = {upgrade.get("upgrade_id"): upgrade for upgrade in ALL_UPGRADES}

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Constants

# COMMAND ----------

# Load current release version so that we can load the proper model number
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data

# COMMAND ----------

# BuildingSet metadata
building_metadata_table_path = get_building_metadata_table(BUILDING_SET)
building_set_metadata = spark.table(building_metadata_table_path)

# Retrieve heat pump sizing
if BUILDING_SET in [BuildingSet.RESSTOCK_2024_2, BuildingSet.RESSTOCK_2022_1]:
    building_set_sizing = spark.table("building_model.resstock_ra_outputs_annual")
elif BUILDING_SET == BuildingSet.RESSTOCK_2025_1:
    raise ValueError(f"Annual outputs for BuildingSet {BuildingSet.RESSTOCK_2025_1} have not been calculated.")
else:
    building_set_sizing = spark.table(f"processed.{BUILDING_SET}.resstock_ra_outputs_annual")
building_set_sizing = building_set_sizing.filter(F.col("building_set") == BUILDING_SET).select(
    "building_id",
    "upgrade_id",
    (F.col("out_hvac_capacity_cooling_kw") * C.KILOWATT_TO_TON_REFRIGERATION).alias("cooling_capacity_tons"),
    (F.col("out_hvac_capacity_heating_kw") * C.KILOWATT_TO_TON_REFRIGERATION).alias("heating_capacity_tons"),
)
# Filter to only ASHP upgrades
heat_pump_upgrades = [
    upgrade
    for upgrade in ALL_UPGRADES
    if (
        upgrade.get("heat_pump")
        and
        # Drop GSHP upgrades
        upgrade.get("upgrade_id") not in [14.01, 14.02, 14.03]
    )
]
heat_pump_upgrade_ids = [heat_pump_upgrade["upgrade_id"] for heat_pump_upgrade in heat_pump_upgrades]
building_set_sizing = building_set_sizing.filter(F.col("upgrade_id").isin(heat_pump_upgrade_ids))

if building_set_sizing.isEmpty():
    dbutils.notebook.exit("ASHPs not modeled for this building set. Skipping prediction.")

# Climate zones
climate_zones = (
    spark.table("geographic_crosswalk.resstock_county_to_other_geographies")
    .select("in_ashrae_iecc_climate_zone_2004")
    .distinct()
)

# RS Means data
rsmeans = spark.table("upfront_cost.rsmeans_county")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load trained model

# COMMAND ----------

# Initialize the MLflow client
client = MlflowClient()

# Get the experiment information
# First establish folder where shared models are stored
experiment_name = "/Shared/upfront_cost_ashp_model"
experiment = client.get_experiment_by_name(experiment_name)
# Load Experiment ID of most the folder
experiment_id = experiment.experiment_id

# Get the most recent version of the model by looking for the current release version tag in the given experiment
run_name = f"ashp_model_{RELEASE_VERSION}"
runs = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")

# Search through runs to find id for this particular model run
run = runs[0]
run_id = run.info.run_id

# Load model based on Run ID
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pre-process single-family & small multifamily metadata and climate zone data

# COMMAND ----------

# Extract letter and number from climate zone data
climate_zones = (
    climate_zones.select("in_ashrae_iecc_climate_zone_2004")
    .withColumn(
        "climate_zone_number",
        F.substring(F.col("in_ashrae_iecc_climate_zone_2004"), 1, 1).astype("int"),
    )
    .withColumn(
        "climate_zone_letter",
        F.substring(F.col("in_ashrae_iecc_climate_zone_2004"), 2, 1),
    )
)

# COMMAND ----------

# Pull Relevant metadata for each occupied single-family & small multifamily building that we'll be predicting for
building_set_metadata_sf_small_mf = (
    building_set_metadata.select(
        "building_id",
        "in_sqft",
        "in_sqft_bin",
        "in_ashrae_iecc_climate_zone_2004",
        "county_geoid",
        "in_weather_file_city",
        "in_state",
        "in_hvac_has_ducts",
    )
    # Filter to occupied SF / small MF buildings
    .filter(
        (
            F.col("in_geometry_building_type_acs").isin(
                ["Single-Family Attached", "Single-Family Detached", "2 Unit", "3 or 4 Unit"]
            )
        )
        & (~F.col("in_is_vacant"))
    )
    # Convert some column names for prediction purposes
    .withColumnsRenamed({"in_sqft": "sqft", "in_hvac_has_ducts": "ducted"})
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the dataframe of features for prediction
# MAGIC
# MAGIC There are 7 features currently used by the model to predict ASHP upfront costs. We will do necessary preprocessing for each feature in turn.

# COMMAND ----------

# Identify which columns will be used for prediction
columns_for_prediction = [
    "heating_capacity_tons",
    "sqft",
    "seer",
    "hspf_region_iv",
    "had_duct_replacement",
    "had_panel_upgrade",
    "ducted",
]

# Instantiate feature table by joining the building metadata with the heat pump sizing table.
# Note that this filters out buildings which did not receive ASHP upgrades.
heat_pump_features = building_set_metadata_sf_small_mf.join(building_set_sizing, on="building_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Panel upgrades and duct replacement

# COMMAND ----------

# Assume ASHP installations for all buildings do not require duct replacement or a panel upgrade
heat_pump_features = heat_pump_features.withColumn("had_duct_replacement", F.lit(False)).withColumn(
    "had_panel_upgrade", F.lit(False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Heating capacity
# MAGIC Heating capacity is required feature for our ASHP quantile regression model. For homes with heating, simulation outputs generally include a heating capacity value we can use directly. However, in unusual cases where it is missing (e.g., upgrades 11.08 and 11.09 for ResStock 2024.2) and for homes without heating, we use the following logic and a heuristic relating cooling capacity, climate zone, and heating capacity:
# MAGIC
# MAGIC - If cooling capacity is available and nonzero in simulation outputs:
# MAGIC     - Estimate heating capacity using the heuristic, known cooling capacity, and climate zone.
# MAGIC - Else:
# MAGIC     - Estimate cooling capacity using a hard-coded list by climate zone.
# MAGIC     - Estimate heating capacity using the heuristic, estimated cooling capacity, and climate zone.

# COMMAND ----------

# Create a table of cooling capacities for each climate zone
cooling_sizing_list = [
    (1, 1.9),
    (2, 1.9),
    (3, 1.85),
    (4, 1.75),
    (5, 1.7),
    (6, 1.65),
    (7, 1.65),
]
cooling_size_df = spark.createDataFrame(cooling_sizing_list, ["climate_zone_number", "cooling_tons_per_ksqft"]).join(
    climate_zones, on="climate_zone_number"
)
heat_pump_features = (
    heat_pump_features.join(cooling_size_df, on="in_ashrae_iecc_climate_zone_2004")
    # create is cold climate feature
    .withColumn("is_cold_climate", F.col("climate_zone_number") > 4)
    # Estimate cooling capacity, if it is Null or 0. This is the case for homes without cooling
    # and for some upgrades we directly import from NREL which do not supply cooling capacity as an output value.
    .withColumn("cooling_capacity_tons_estimated", F.col("sqft") / 1000 * F.col("cooling_tons_per_ksqft"))
    # Impute missing and 0-value cooling capacities with estimates
    .withColumn(
        "cooling_capacity_tons",
        F.when(
            (F.col("cooling_capacity_tons").isNull()) | (F.col("cooling_capacity_tons") == 0),
            F.col("cooling_capacity_tons_estimated"),
        ).otherwise(F.col("cooling_capacity_tons")),
    )
    # Estimate heating capacity using imputed cooling capacity as a proxy
    # (cooling_capacity_tons is either the simulation value or the sqft-based estimate from above)
    .withColumn("heating_capacity_tons_estimated", F.col("cooling_capacity_tons"))
    # Now oversize by 1 ton in cold, dry climates
    .withColumn(
        "heating_capacity_tons_estimated",
        F.when(
            (F.col("climate_zone_letter") == "B") & (F.col("is_cold_climate")),
            F.col("heating_capacity_tons_estimated") + 1,
        ).otherwise(F.col("heating_capacity_tons_estimated")),
    )
    # Impute missing and 0-values with estimated values
    .withColumn(
        "heating_capacity_tons",
        F.when(
            (F.col("heating_capacity_tons").isNull()) | (F.col("heating_capacity_tons") == 0),
            F.col("heating_capacity_tons_estimated"),
        ).otherwise(F.col("heating_capacity_tons")),
    )
    # Drop columns no longer needed
    .drop(
        "climate_zone_number",
        "cooling_tons_per_ksqft",
        "cooling_capacity_tons",
        "cooling_capacity_tons_estimated",
        "heating_capacity_tons_estimated",
        "is_cold_climate",
    )
)

# COMMAND ----------

# Check: make sure we imputed fully
assert heat_pump_features.filter(
    (F.col("heating_capacity_tons").isNull()) | (F.col("heating_capacity_tons") == 0)
).isEmpty(), "Imputation error: null or zero heating capacities"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### HSPF & SEER
# MAGIC HSPF & SEER values are associated with the ASHP associated with a given upgrade applied to a given building.

# COMMAND ----------

# Create mapping expression for retrieving heat pump
heat_pump_mapping_expr = F.create_map(
    [F.lit(x.get("heat_pump") if isinstance(x, dict) else x) for x in chain(*ALL_UPGRADES_BY_ID.items())]
)
heat_pump_features = heat_pump_features.withColumn("appliance_option", heat_pump_mapping_expr[F.col("upgrade_id")])

# Use mapping expressions to retrieve HSPF and SEER values for ducted and ductless heat pumps based on the heat pump
for rating in ["seer_ducted", "seer_ductless", "hspf_ducted", "hspf_ductless"]:
    map_expr = F.create_map(
        [F.lit(x[rating] if isinstance(x, dict) else x) for x in chain(*HEAT_PUMPS_BY_NAME.items())]
    )
    heat_pump_features = heat_pump_features.withColumn(
        rating,
        F.when(F.col("appliance_option").isNull(), F.lit(None)).otherwise(map_expr[F.col("appliance_option")]),
    )

# Use ducting information to determine whether to use ducted or ductless HSPF and SEER values
heat_pump_features = (
    heat_pump_features.withColumn(
        "seer", F.when(F.col("ducted"), F.col("seer_ducted")).otherwise(F.col("seer_ductless"))
    )
    .withColumn("hspf_region_iv", F.when(F.col("ducted"), F.col("hspf_ducted")).otherwise(F.col("hspf_ductless")))
    # In some cases buildings with ducting are eligible for upgrades involving ductless-only ASHPs.
    # For these cases, use the ductless SEER/HSPF values.
    .withColumn(
        "seer",
        F.when(
            (F.col("appliance_option").isNotNull()) & (F.col("ducted")) & (F.col("seer").isNull()),
            F.col("seer_ductless"),
        ).otherwise(F.col("seer")),
    )
    .withColumn(
        "hspf_region_iv",
        F.when(
            (F.col("appliance_option").isNotNull()) & (F.col("ducted")) & (F.col("hspf_region_iv").isNull()),
            F.col("hspf_ductless"),
        ).otherwise(F.col("hspf_region_iv")),
    )
    # Drop intermediate columns
    .drop("seer_ducted", "seer_ductless", "hspf_ducted", "hspf_ductless")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict upfront costs for each building ID

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Predict on the dataframe of building energy model features

# COMMAND ----------

# Convert to pandas to allow for quantile predictions
prediction_df = heat_pump_features.select(
    "building_id", "upgrade_id", "appliance_option", *columns_for_prediction
).toPandas()

# COMMAND ----------

# Predict at preset quantiles
predictions = (
    loaded_model.predict_quantiles(prediction_df[columns_for_prediction], alpha=[0.2, 0.5, 0.8])
    .droplevel(0, axis=1)
    .rename(columns={0.2: "upfront_cost_20percentile", 0.5: "upfront_cost_median", 0.8: "upfront_cost_80percentile"})
)
predictions = pd.concat([prediction_df[["building_id", "upgrade_id", "appliance_option"]], predictions], axis=1)
# Convert to spark for re-joining
predictions = spark.createDataFrame(predictions)
# Re-join with the original dataframe
predictions = heat_pump_features.join(predictions, on=["building_id", "upgrade_id", "appliance_option"], how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Adjust predicted prices to geographic area of BuildingSet buildings using RSMeans

# COMMAND ----------

# Scale the predicted price by the RSMeans factor so that we are localizing costs to each building's location
predictions_with_rsmeans = predictions.join(
    rsmeans.where((F.col("category") == "fire suppression, plumbing & hvac") & (F.col("component") == "total")).select(
        "county_geoid", "rsmeans_price_index"
    ),
    on="county_geoid",
)

for column in ["upfront_cost_20percentile", "upfront_cost_median", "upfront_cost_80percentile"]:
    predictions_with_rsmeans = predictions_with_rsmeans.withColumn(
        column, F.col(column) * F.col("rsmeans_price_index") / 100
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write out to tables

# COMMAND ----------

# Specify BuildingSet in a table column and use selective overwrite for national BuildingSets.
predictions_with_rsmeans = predictions_with_rsmeans.withColumn("building_set", F.lit(BUILDING_SET))
write_options = {}
if BUILDING_SET in national_building_metadata:
    write_options = {"overwriteSchema": "false", "replaceWhere": f"building_set = '{BUILDING_SET}'"}

# Determine output table name based on building set
db_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="processed",
    schema="upfront_costs",
    table_name="ashp_upfront_cost_by_building_appliance_upgrade",
)

# Write out table
sparkdb.write_delta_table(
    df=predictions_with_rsmeans,
    db_table_name=db_table_name,
    mode=MODE,
    release_version=RELEASE_VERSION,
    **write_options,
)
