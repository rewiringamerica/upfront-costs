# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Generate Features to Train Solar Upfront Cost Model
# MAGIC
# MAGIC ### Goal
# MAGIC Create the features that will be used to train the solar upfront cost model by levelizing costs based on
# MAGIC time (using construction price index) and location (based on RSMeans Price Index).
# MAGIC
# MAGIC ### Process
# MAGIC Load LBNL cost of installation data. In order to levelize cost data, merge in Construction Inflation index by
# MAGIC year/month of installation and RSMeans data on geographic installation costs by zip code. Adjust the LBNL
# MAGIC installation costs based on indices in order to levelize to 'national' cost for each data point. Use these
# MAGIC levelized numbers as training data for a quantile regression model.
# MAGIC
# MAGIC ### Inputs
# MAGIC - `processed.upfront_costs.lbnl_rooftop_solar`
# MAGIC - `upfront_cost.construction_inflation`
# MAGIC - `upfront_cost.rsmeans_zip`
# MAGIC
# MAGIC ### Output
# MAGIC - `processed.upfront_costs.solar_training_features`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup

# COMMAND ----------

# imports
import pyspark.sql.functions as F
from pyspark.sql.functions import col

# internal
from dmlutils import sparkdb

from src import util

# COMMAND ----------

# Set I/O mode
dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

# COMMAND ----------

# Load current release version to version feature table written at end of script
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load the Data

# COMMAND ----------

# Load solar cost data from Lawrence Berkeley
solar_costs_df = spark.table(sparkdb.modify_db_table_name("processed.upfront_costs.lbnl_rooftop_solar", MODE))

# Load construction inflation prices to levelize solar costs over time
construction_inflation_df = spark.table(sparkdb.modify_db_table_name("upfront_cost.construction_inflation", MODE))

# Load RSMeans data by zip code to levelize solar costs over geographies
rsmeans_zip_df = spark.table(sparkdb.modify_db_table_name("upfront_cost.rsmeans_zip", MODE))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Training Features

# COMMAND ----------

# Subset RSMeans data to electrical work, the best proxy for solar installation since there is not specific solar
# data with RSMeans
rsmeans_zip_df = rsmeans_zip_df.filter(
    (col("component") == "total") & (col("category") == "electrical, communications & util.")
)

# COMMAND ----------

# # Subset solar data to only those entries from 2021 and beyond. The LBNL data goes back further but costs have fallen
# so dramatically in that period that it would likely overestimate costs to use all the data. Costs below $100 are
# suspect and are therefore filtered out
solar_costs_df = (
    solar_costs_df.withColumn("total_installed_price", F.col("total_installed_price").cast("double"))
    .filter((col("year") > 2020) & (F.col("total_installed_price") > 100))
    # To avoid the outsized impact of outliers, and given that most residential rooftop solar
    # installations are <10kw, we conservatively filter out installations >= 500 kW
    .filter((col("system_size_kw") < 500))
    # Manual exploration of the data suggests that installations > $150/watt are serious outliers
    # and should be removed. For comparison, the median is roughly $2.40/watt.
    .filter((col("cost_dollars_per_watt_installed") < 150))
    .select("zip_code", "month", "year", "system_size_kw", "total_installed_price")
)

# COMMAND ----------

# Merge in construction and inflation indices
merged_df = (
    solar_costs_df
    # RS Means zip code only at 3 digit zip code level (not usual 5 digit) so create column to join on
    .withColumn("zip_code_prefix", F.substring(col("zip_code"), 1, 3))
    # Merge in construction inflation index on Month and Year combination
    .join(construction_inflation_df, how="left", on=["month", "year"])
    # Merge in RS Means geographic index on recently created zip prefix column
    .join(rsmeans_zip_df, how="left", on="zip_code_prefix")
)

# COMMAND ----------

# use the indices to levelize the cost of installation by time and geography
levelized_solar_cost_df = merged_df.withColumn(
    "levelized_total_installed_price_dollars", F.col("total_installed_price") / (col("construction_price_index") / 100)
).withColumn(
    "levelized_total_installed_price_dollars",
    F.col("levelized_total_installed_price_dollars") / (col("rsmeans_price_index") / 100),
)

# COMMAND ----------

# Select the columns that are relevant for this model
training_features = levelized_solar_cost_df.select("system_size_kw", "levelized_total_installed_price_dollars")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Save Feature Table

# COMMAND ----------

# Save out features
sparkdb.write_delta_table(
    df=training_features,
    db_table_name="processed.upfront_costs.solar_training_features",
    mode=MODE,
    release_version=RELEASE_VERSION,
)
