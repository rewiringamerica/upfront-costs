# Databricks notebook source
# MAGIC %md
# MAGIC ## Ground Source Heat Pump Upfront cost Model Training
# MAGIC ### Goal
# MAGIC Training the XGBoost regressor on upfront cost data to build a model which we can then use
# MAGIC to predict ground source heat pump costs for ResStock building models.
# MAGIC
# MAGIC ### Process
# MAGIC * Read gshp training features
# MAGIC * Use MLFlow to train XGBoost model with squared error objective (**Note**: this requires a
# MAGIC notebook with MLFlow and xgboost packages already installed or you can use cell 3 to `pip install` those)
# MAGIC * Save best model parameters in the experiments
# MAGIC
# MAGIC ### Inputs
# MAGIC * `processed.upfront_costs.gshp_training_features`
# MAGIC
# MAGIC ### Outputs
# MAGIC * None. Saves best model training parameters in the experiments with
# MAGIC `experiment_name = "Shared/upfront_cost_gshp"`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Packages and Establish Constants

# COMMAND ----------

# DBTITLE 1,Python packages Imports
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import make_scorer, median_absolute_error
from xgboost import XGBRegressor

from dmlutils import sparkdb

from src import modeling_util, util

# COMMAND ----------

# DBTITLE 1,Dropdown and text widgets in databricks utilities
dbutils.widgets.dropdown("mode", "feature", ["feature", "main"])
MODE = dbutils.widgets.get("mode")

# COMMAND ----------

# Load current release version to version feature table written at end of script
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load data

# COMMAND ----------

# DBTITLE 1,Read in training features table
training_features = spark.table(sparkdb.modify_db_table_name("processed.upfront_costs.gshp_training_features", MODE))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create feature table

# COMMAND ----------

# DBTITLE 1,Create training set and transfer to pandas
price_gshp = training_features.select("building_sq_ft", "total_capacity_tons", "adjusted_system_cost").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initialize the model helper

# COMMAND ----------

# DBTITLE 1,Define features and target, set experiment location, initialize model helper module
features = ["building_sq_ft", "total_capacity_tons"]
target = "adjusted_system_cost"

# initialize ModelHelper class
experiment_name = "/Shared/upfront_cost_gshp"

helper = modeling_util.ModelHelper(price_gshp, features, target, experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train XGBoost model

# COMMAND ----------

# DBTITLE 1,Define hyperparameters and train the model
param_grid = {
    "max_depth": [3, 7, 10, 12],
    "eta": [0.2, 0.1, 0.05, 0.03],
    "lambda": [0, 1, 3, 5, 7],
    "subsample": [0.5, 0.7, 1],
}

model = XGBRegressor(random_state=42)  # use xgb regressor with squared error objective
# Define scorer
scorer = make_scorer(median_absolute_error, greater_is_better=False)

helper.train_model(model, scorer, param_grid)  # train model using hyperparameters
helper.log_model(f"gshp_model_{RELEASE_VERSION}")  # log best model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Histogram of errors

# COMMAND ----------

# DBTITLE 1,Errors histogram
# fetch the experiment
experiment = mlflow.get_experiment_by_name(experiment_name)

# retrieve all runs in the experiment
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# filter the runs to get the run ID with the name 'xgboost'
xgboost_run = runs[runs["tags.mlflow.runName"] == "gshp_xgboost"].iloc[0]
run_id = xgboost_run["run_id"]

# set the path to download the artifact
artifact_path = "predictions.csv"
local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

# load the CSV file into a DataFrame
predictions_df = pd.read_csv(local_path)

# plot the histogram of the "errors" column
plt.figure(figsize=(10, 6))
plt.hist(predictions_df["error"], bins=30, edgecolor="k", alpha=0.7)
plt.title("Histogram of Errors")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
