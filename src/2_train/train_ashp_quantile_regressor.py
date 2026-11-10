# Databricks notebook source
# MAGIC %md
# MAGIC ## Train Quantile Regressor on ASHP Data
# MAGIC ### Goal
# MAGIC Training a quantile regression model on upfront cost data which we can then use to predict
# MAGIC heat pump costs for ResStock building models.
# MAGIC
# MAGIC ### Process
# MAGIC - Create a training set from the data created in the `generate_ashp_training_features` script.
# MAGIC - Initialize the model helper, train the quantile regression model
# MAGIC
# MAGIC ### Inputs
# MAGIC - `processed.upfront_costs.ashp_training_features` : includes target data to create a training set
# MAGIC
# MAGIC ### Outputs
# MAGIC - Logged model in MLFlow tagged with current release version

# COMMAND ----------

# MAGIC %md ### Set Up

# COMMAND ----------

# DBTITLE 1,Python Imports
# imports
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import make_scorer, median_absolute_error

# Here we're using skpro because `sklearn.linear_model.QuantileRegressor`
# doesn't allow for multiple quantiles within a single `.fit()` call
from skpro.regression.multiquantile import MultipleQuantileRegressor

from dmlutils import sparkdb

# internal
from src import modeling_util, util

# COMMAND ----------

# DBTITLE 1,Set I/O mode
dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

# COMMAND ----------

# MAGIC %md #### Create Constants

# COMMAND ----------

# Load current release version to properly load training data and save model after training
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create feature table

# COMMAND ----------

# Read the data that contains both features and targets
training_df = spark.table(sparkdb.modify_db_table_name("processed.upfront_costs.ashp_training_features", MODE))

# COMMAND ----------

# Load training data to a Pandas DF
training_pd = training_df.toPandas()

# Convert several columns to numeric
columns_to_convert = ["project_cost_adjusted", "sqft"]
# Convert necessary columns to numeric
for column in columns_to_convert:
    training_pd[column] = pd.to_numeric(training_pd[column])

# Convert boolean features to numeric, they are currently stored as objects because of nan
training_pd["ducted"] = np.where(
    training_pd["ducted"] == True,  # noqa: E712
    1,
    np.where(training_pd["ducted"] == False, 0, np.nan),  # noqa: E712
)
training_pd["had_duct_replacement"] = np.where(
    training_pd["had_duct_replacement"] == True,  # noqa: E712
    1,
    np.where(training_pd["had_duct_replacement"] == False, 0, np.nan),  # noqa: E712
)
training_pd["had_panel_upgrade"] = np.where(
    training_pd["had_panel_upgrade"] == True,  # noqa: E712
    1,
    np.where(training_pd["had_panel_upgrade"] == False, 0, np.nan),  # noqa: E712
)

# filter out square footage and project costs that are considered to be outliers.
training_pd = training_pd[
    (training_pd["sqft"] > 100)
    & (training_pd["sqft"] <= 6000)
    & (training_pd["project_cost_adjusted"] <= 60000)
    & (training_pd["project_cost_adjusted"] > 0)
]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initialize the model helper

# COMMAND ----------

# Define features and target to use in ModelHelper initialization
features = [
    "heating_capacity_tons",
    "sqft",
    "seer",
    "hspf_region_iv",
    "had_duct_replacement",
    "had_panel_upgrade",
    "ducted",
]
target = "project_cost_adjusted"

# Filter training data so that it only contains features and targets
training_pd = training_pd[features + [target]]
# Drop NA values
print(f"Row count w/ NAs: {training_pd.shape[0]}")
training_pd = training_pd.dropna(axis=0)
print(f"Row count w/o NAs: {training_pd.shape[0]}")

# Initialize the ModelHelper class
helper = modeling_util.ModelHelper(training_pd, features, target, experiment_name="/Shared/upfront_cost_ashp_model")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train quantile regression model

# COMMAND ----------

# Define hyperparameter distributions for the QuantileRegressor
param_grid = {
    # vary L1 regularization param (`alpha`)
    # NOTE: in this implementation, L1 values are applied to all quantiles
    # for a given MultipleQuantileRegressor. We may want more flexibility in the future.
    "quantile_regressor": [
        QuantileRegressor(solver="highs", alpha=0.001),  # solver="highs" is default
        QuantileRegressor(solver="highs", alpha=0.01),
        QuantileRegressor(solver="highs", alpha=0.1),
        QuantileRegressor(solver="highs", alpha=1),
        QuantileRegressor(solver="highs", alpha=5),
    ]
}

# Instantiate model object
model = MultipleQuantileRegressor(
    quantile_regressor=QuantileRegressor(solver="highs", alpha=1),
    # Define quantiles (also coincidentally called `alpha` by skpro)
    alpha_name="quantile",
    alpha=[0.20, 0.5, 0.80],
    n_jobs=3,
    # Enforce monotonicity
    sort_quantiles=True,
)
# Define scorer
scorer = make_scorer(median_absolute_error, greater_is_better=False)

# COMMAND ----------

# Train the model, saving hyperparameter search results as we go.
helper.train_model(model, scorer, param_grid, cv=5, n_iter=5)

# COMMAND ----------

# Log the best scoring model (averaged across CV folds)
helper.log_model(f"ashp_model_{RELEASE_VERSION}")
