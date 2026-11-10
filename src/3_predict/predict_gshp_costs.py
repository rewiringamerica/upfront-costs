# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Predict Ground Source Heat Pump (GSHP) Costs on a BuildingSet
# MAGIC
# MAGIC ### Goal
# MAGIC To predict the upfront cost of ground source heat pumps (GSHPs) at the building level for all
# MAGIC occupied single-family buildings in a given BuildingSet. Note: if GSHPs were not modeled for the given
# MAGIC BuildingSet, the notebook will exit.
# MAGIC
# MAGIC ### Process
# MAGIC We first create a table of the features on which we want to predict the upfront cost of a GSHP. These
# MAGIC include 1) square footage (we use the median square footage in each bucket) and 2) heat pump size in tons.
# MAGIC
# MAGIC We have modeled sizing for GSHPs according to available sizing methodologies,
# MAGIC assuming both a setpoint setback and no setpoint setback. To include this, we will create a df with the median
# MAGIC sizing (drawn from the RAStock annual outputs (i.e., `building_model.resstock_ra_outputs_annual`)) across
# MAGIC buckets defined by `(in_ashrae_iecc_climate_zone_2004, in_sqft_bin)`. We will then join this DF to the
# MAGIC `heat_pump_features` df created using the logic described above.
# MAGIC
# MAGIC We then predict costs for all occupied single-family buildings using the gradient boosted trees (XGBoost) model trained in
# MAGIC `train_gshp_xgboost_regressor.py`.
# MAGIC
# MAGIC ### I/O
# MAGIC ##### Inputs:
# MAGIC - `upfront_cost.rsmeans_county`
# MAGIC - BuildingSet metadata
# MAGIC - `processed.<building_set>.resstock_ra_outputs_annual`
# MAGIC - ML Experiment trained and stored at: `Shared/upfront_cost_gshp`
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `processed.<building_set>.gshp_upfront_cost_by_building_upgrade_appliance`
# MAGIC Note that buildings may have between 1-3 GSHP options modeled in this table due to the sizing methodology described above.
# MAGIC
# MAGIC ##### TO DO:
# MAGIC - Consider adding 20th & 80th percentile prediction estimates at the building level
# MAGIC

# COMMAND ----------

# MAGIC %md ### Import Packages and Establish Constants

# COMMAND ----------

# DBTITLE 1,Import packages
from itertools import chain

import mlflow
import pyspark.sql.functions as F
from mlflow.tracking import MlflowClient

from dmlutils import constants as C
from dmlutils import sparkdb
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

dbutils.widgets.dropdown("mode", "feature", ["feature", "main"])
MODE = dbutils.widgets.get("mode")

dbutils.widgets.dropdown(
    "building_set", choices=[b.value for b in BuildingSet], defaultValue=BuildingSet.RESSTOCK_2024_2
)
BUILDING_SET = dbutils.widgets.get("building_set")

# If using a regional BuildingSet, use the list of regional upgrades; otherwise, use the full list of upgrades
ALL_UPGRADES = REGIONAL_UPGRADES.get(BUILDING_SET, UPGRADES)
ALL_UPGRADES_BY_ID = {upgrade.get("upgrade_id"): upgrade for upgrade in ALL_UPGRADES}

# COMMAND ----------

# Load current release version to version feature table written at end of script
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md ### Load Data

# COMMAND ----------

# DBTITLE 1,Read in BuildingSet metadata, sizing, and RSMeans
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
    (F.col("out_hvac_capacity_heating_kw") * C.KILOWATT_TO_TON_REFRIGERATION).alias("total_capacity_tons"),
)
# Filter to only GSHP upgrades
# NOTE: this may fail if we create new GSHP upgrades or new regional versions of existing GSHP upgrades
building_set_sizing = building_set_sizing.filter(F.col("upgrade_id").isin([14.01, 14.02, 14.03]))

if building_set_sizing.isEmpty():
    dbutils.notebook.exit("GSHPs not modeled for this building set. Skipping prediction.")

# Climate zones

# RS Means data
rsmeans = spark.table("upfront_cost.rsmeans_county")

# COMMAND ----------

# DBTITLE 1,Single-family metadata and median square footage by size
building_set_metadata_sf = building_set_metadata.select(
    "building_id",
    "in_sqft",
    "in_sqft_bin",
    "in_ashrae_iecc_climate_zone_2004",
    "county_geoid",
    "in_weather_file_city",
    "in_state",
).filter(
    # Filter to occupied SF buildings
    (F.col("in_geometry_building_type_acs").isin(["Single-Family Attached", "Single-Family Detached"]))
    & (~F.col("in_is_vacant"))
)

# COMMAND ----------

# DBTITLE 1,Preprocess sizing table
# Create mapping expression for retrieving heat pump
heat_pump_mapping_expr = F.create_map(
    [F.lit(x.get("heat_pump") if isinstance(x, dict) else x) for x in chain(*ALL_UPGRADES_BY_ID.items())]
)

building_set_sizing = building_set_sizing.withColumn("appliance_option", heat_pump_mapping_expr[F.col("upgrade_id")])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create dataframe of features for prediction

# COMMAND ----------

# DBTITLE 1,Join some geoid and sqft bin to sizing dataset
# Step 1: Join sizing to building_set_metadata_sf on building_id
# Step 2: Add rsmeans price index joining to rsmeans df on county-geoid
# Step 3: Select building sq ft, sq ft bin, county geoid, rsmeans price index, heating capacity, and appliance option

prediction_features = (
    building_set_metadata_sf.join(building_set_sizing, on="building_id")  # 1
    .join(  # 2
        rsmeans.where(
            (F.col("category") == "fire suppression, plumbing & hvac") & (F.col("component") == "total")
        ).select("county_geoid", "rsmeans_price_index"),
        on="county_geoid",
    )
    .select(  # 3
        "building_id",
        F.col("in_sqft").alias("building_sq_ft"),
        "in_sqft_bin",
        "rsmeans_price_index",
        "total_capacity_tons",
        "upgrade_id",
        "appliance_option",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict on feature dataframe using the trained model in the MLFlow experiment

# COMMAND ----------

# DBTITLE 1,Load the trained model from MLFlow
# Initialize the MLflow client
client = MlflowClient()

# Get the experiment information
experiment_name = "/Shared/upfront_cost_gshp"
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Get the most recent version of the model by looking for the current release version tag in the given experiment
run_name = f"gshp_model_{RELEASE_VERSION}"
runs = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")

# Search through runs to find id for this particular model run
run = runs[0]
run_id = run.info.run_id

# Load model based on Run ID
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")

# COMMAND ----------

# DBTITLE 1,Predict using the loaded model
columns_for_prediction = ["building_sq_ft", "total_capacity_tons"]

# Predict with the loaded model and established columns for prediction features
gshp_predictions = prediction_features.withColumn(
    "prediction", loaded_model(F.struct(*[F.col(column) for column in columns_for_prediction]))
)

# adjust predicted prices to building locations using RSmeans price index
adjusted_gshp_predictions = gshp_predictions.withColumn(
    "upfront_cost_median", F.col("prediction") * (F.col("rsmeans_price_index") / 100)
).drop("rsmeans_price_index", "prediction")

# COMMAND ----------

# DBTITLE 1,Get 20th and 80th percentile values
# TODO: Consider methodology for building-level percentile estimates
gshp_upfront_costs = adjusted_gshp_predictions.withColumn(
    "upfront_cost_20percentile", F.lit(None).cast("double")
).withColumn("upfront_cost_80percentile", F.lit(None).cast("double"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write out to table

# COMMAND ----------

# Specify BuildingSet in a table column and use selective overwrite for national BuildingSets.
gshp_upfront_costs = gshp_upfront_costs.withColumn("building_set", F.lit(BUILDING_SET))
write_options = {}
if BUILDING_SET in national_building_metadata:
    write_options = {"overwriteSchema": "false", "replaceWhere": f"building_set = '{BUILDING_SET}'"}

# Determine output table name based on building set
db_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="processed",
    schema="upfront_costs",
    table_name="gshp_upfront_cost_by_building_upgrade_appliance",
)

# Write out table
sparkdb.write_delta_table(
    df=gshp_upfront_costs,
    db_table_name=db_table_name,
    mode=MODE,
    release_version=RELEASE_VERSION,
    **write_options,
)
