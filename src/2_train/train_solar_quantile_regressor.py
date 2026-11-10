# Databricks notebook source
# MAGIC %md
# MAGIC ## Train Quantile Regressor on Solar Data
# MAGIC ### Goal
# MAGIC Training a quantile regression model on solar upfront cost data which we can then use to predict
# MAGIC rooftop solar costs for ResStock building models.
# MAGIC
# MAGIC ### Process
# MAGIC - Create a training set from the data created in the `generate_solar_training_features` script.
# MAGIC - Initialize the model helper, train the quantile regression model
# MAGIC
# MAGIC ### Inputs
# MAGIC - `processed.upfront_costs.solar_training_features` : includes target data to create a training set
# MAGIC
# MAGIC ### Outputs
# MAGIC - Logged model in MLFlow tagged with current release version

# COMMAND ----------

# MAGIC %md ### Set Up

# COMMAND ----------

# DBTITLE 1,Python Imports
# imports
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
training_df = spark.table(sparkdb.modify_db_table_name("processed.upfront_costs.solar_training_features", MODE))

# COMMAND ----------

# Load training data to a Pandas DF
training_pd = training_df.toPandas()

# Convert several columns to numeric
columns_to_convert = ["system_size_kw", "levelized_total_installed_price_dollars"]
# Convert necessary columns to numeric
for column in columns_to_convert:
    training_pd[column] = pd.to_numeric(training_pd[column])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initialize the model helper

# COMMAND ----------

# Define features and target to use in ModelHelper initialization
features = ["system_size_kw"]
target = "levelized_total_installed_price_dollars"

# Filter training data so that it only contains features and targets
training_pd = training_pd[features + [target]]
# Drop NA values
print(f"Row count w/ NAs: {training_pd.shape[0]}")
training_pd = training_pd.dropna(axis=0)
print(f"Row count w/o NAs: {training_pd.shape[0]}")

# Solving the quantile regressor's linear program on 160k samples
# is extremely slow. To save time we randomly sample the training data.
training_pd_sampled = training_pd.sample(n=40000, random_state=42)

# Initialize the ModelHelper class
helper = modeling_util.ModelHelper(
    training_pd_sampled, features, target, experiment_name="/Shared/upfront_cost_solar_model"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train quantile regression model

# COMMAND ----------

# Define hyperparameter distributions for the QuantileRegressor
param_grid = {
    # NOTE: we currently don't perform any hyperparameter exploration due to the
    # simplicity of the model and use of just one feature. This grid object is
    # set up for compatibility if we add more complexity in the future.
    "quantile_regressor": [
        QuantileRegressor(solver="highs", alpha=0),  # solver="highs" is default
    ]
}

# Instantiate model object
model = MultipleQuantileRegressor(
    quantile_regressor=QuantileRegressor(solver="highs", alpha=0),
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

# Train the model
helper.train_model(model, scorer, param_grid, cv=5, n_iter=1)

# COMMAND ----------

# Log the best scoring model (averaged across CV folds)
helper.log_model(f"solar_model_{RELEASE_VERSION}")
