# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Predict Solar Upfront Costs on a BuildingSet
# MAGIC
# MAGIC ### Goal
# MAGIC Run the trained Solar Cost model on the data for each building in a given BuildingSet and every modeled solar config.
# MAGIC This script allows Research to use Solar Upfront Costs in downstream analyses like the affordability.
# MAGIC
# MAGIC ### I/O
# MAGIC ##### Inputs:
# MAGIC - `processed.<building_set>.solar_configuration_by_building_solarconfig`
# MAGIC - `upfront_cost.rsmeans_county`
# MAGIC - Pretrained Quantile Regressor Model tagged with current release version number
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `processed.<building_set>.solar_upfront_cost_by_building_solarconfig`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import pandas as pd
import pyspark.sql.functions as F
from mlflow.tracking import MlflowClient

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
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

dbutils.widgets.dropdown(
    "building_set", choices=[b.value for b in BuildingSet], defaultValue=BuildingSet.RESSTOCK_2024_2
)
BUILDING_SET = dbutils.widgets.get("building_set")

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

# BuildingSet Metadata
building_metadata_table_path = get_building_metadata_table(BUILDING_SET)
building_set_metadata = spark.table(building_metadata_table_path)

# RS Means data
rsmeans = spark.table("upfront_cost.rsmeans_county")

# COMMAND ----------

# Load in buildings with solar configurations
solar_config_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="processed",
    schema="der",
    table_name="solar_configuration_by_building_solarconfig",
    mode=MODE,
)
building_level_solar_configurations = spark.table(solar_config_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load trained model

# COMMAND ----------

# Initialize the MLflow client
client = MlflowClient()

# Get the experiment information
# First establish folder where shared models are stored
experiment_name = "/Shared/upfront_cost_solar_model"
experiment = client.get_experiment_by_name(experiment_name)
# Load Experiment ID of most the folder
experiment_id = experiment.experiment_id

# Get the most recent version of the model by looking for the current release version tag in the given experiment
run_name = f"solar_model_{RELEASE_VERSION}"
runs = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")

# Search through runs to find id for this particular model run
run = runs[0]
run_id = run.info.run_id

# Load model based on Run ID
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict upfront costs for each building ID

# COMMAND ----------

# Identify which columns will be used for prediction
columns_for_prediction = ["system_size_kw"]

# Drop buildings with missing values for these columns. These are generally homes without roofs in ResStock,
# e.g. multifamily units where the roof is shared between multiple units.
building_level_solar_configurations = building_level_solar_configurations.dropna(subset=columns_for_prediction)

# Convert to pandas to allow for quantile predictions
prediction_df = building_level_solar_configurations.select(
    "building_id", "solar_upgrade_id", *columns_for_prediction
).toPandas()

# COMMAND ----------

# Predict at preset quantiles
predictions = (
    loaded_model.predict_quantiles(prediction_df[columns_for_prediction], alpha=[0.2, 0.5, 0.8])
    .droplevel(0, axis=1)
    .rename(columns={0.2: "upfront_cost_20percentile", 0.5: "upfront_cost_median", 0.8: "upfront_cost_80percentile"})
)
predictions = pd.concat([prediction_df[["building_id", "solar_upgrade_id"]], predictions], axis=1)
# Convert to spark for re-joining
predictions = spark.createDataFrame(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adjust predicted prices to the geographic area of each building using RSMeans

# COMMAND ----------

# Scale the predicted price by the RSMeans factor so that we are localizing costs to each building's location
solar_cost_predictions = (
    building_set_metadata.select("building_id", "county_geoid")
    .join(predictions, on="building_id", how="left")
    .join(
        rsmeans.where((F.col("component") == "total") & (F.col("category") == "electrical, communications & util.")),
        on="county_geoid",
    )
)

for column in ["upfront_cost_20percentile", "upfront_cost_median", "upfront_cost_80percentile"]:
    solar_cost_predictions.withColumn(
        column,
        (F.when(F.col(column).isNull(), F.lit(None)).otherwise(F.col(column) * F.col("rsmeans_price_index") / 100)),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write out to tables

# COMMAND ----------

# Specify BuildingSet in a table column and use selective overwrite for national BuildingSets.
solar_cost_predictions = solar_cost_predictions.withColumn("building_set", F.lit(BUILDING_SET))
write_options = {}
if BUILDING_SET in national_building_metadata:
    write_options = {"overwriteSchema": "false", "replaceWhere": f"building_set = '{BUILDING_SET}'"}

# Determine output table name based on building set
db_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="processed",
    schema="upfront_costs",
    table_name="solar_upfront_cost_by_building_solarconfig",
)

# Write out table
sparkdb.write_delta_table(
    df=solar_cost_predictions,
    db_table_name=db_table_name,
    mode=MODE,
    release_version=RELEASE_VERSION,
    **write_options,
)
