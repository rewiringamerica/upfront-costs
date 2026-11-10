# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Ground Source Heat Pump Upfront Costs Training Features
# MAGIC
# MAGIC ### Goal
# MAGIC Generate training features for our upfront costs model from two datasets:
# MAGIC - [Mass CEC](https://www.masscec.com/public-records-requests)
# MAGIC - [NYSERDA](https://www.notion.so/rewiringamerica/GSHP-Field-Data-d1e0b3255a5847e2951227aa0182959f?pvs=4)
# MAGIC
# MAGIC ### Process
# MAGIC Reads in data and processes those. Attaches location varying costs (RS Means). Saves all into one pricing
# MAGIC table as gshp upfront cost training features. NOTE: unlike other appliances/upgrades, we do not adjust GSHP
# MAGIC costs for inflation. See the Upfront Costs Technical Methodology in Notion for an explanation.
# MAGIC
# MAGIC ### Inputs
# MAGIC - G.RAW_DATA_FPATH/ 'gshp' / 'ResidentialandSmallScaleGSHPProjectDatabase.xlsx'
# MAGIC - G.RAW_DATA_FPATH/ 'gshp' / 'NYSERDA_gshp_upfront_cost.csv'
# MAGIC - `processed.upfront_costs.gshp_geocoded_city_states` (Optional)
# MAGIC
# MAGIC ### Outputs
# MAGIC - `processed.upfront_costs.gshp_training_features`

# COMMAND ----------

# MAGIC %md ### Import Packages and Establish Constants

# COMMAND ----------

# DBTITLE 1,Import packages
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from dmlutils import data_cleaning_utils as dcutils
from dmlutils import sparkdb
from dmlutils.constants import BTU_PER_TON

import src.globals as G
from src import geocoding_util, util

# COMMAND ----------

# DBTITLE 1,Drop-down and text widget in DB utilities
dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
dbutils.widgets.dropdown("re_geocode", "False", ["True", "False"])  # whether to re-geocode the records
MODE = dbutils.widgets.get("mode")
RE_GEOCODE = eval(dbutils.widgets.get("re_geocode"))

# COMMAND ----------

# Load current release version to version feature table written at end of script
RELEASE_VERSION = util.get_release_version()

# Establish API Key
GOOGLE_API_KEY = dbutils.secrets.get(scope="datascience", key="google_geocoding_api_key")

# COMMAND ----------

# MAGIC %md ### Establish Schema & Read-In Data

# COMMAND ----------

# DBTITLE 1,Read NYSERDA upfront cost dataframe
schema = StructType(
    [
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("building sq ft", DoubleType(), True),
        StructField("previous heat distribution", StringType(), True),
        StructField("hp function", StringType(), True),
        StructField("total bore depth feet", DoubleType(), True),
        StructField("total capacity tons", DoubleType(), True),
        StructField("total system cost", DoubleType(), True),
        StructField("database", StringType(), True),
        StructField("number of bores", DoubleType(), True),
    ]
)

# Load NYSERDA data
ny_data = spark.read.csv(str(G.RAW_DATA_FPATH / "gshp" / "NYSERDA_gshp_upfront_cost.csv"), header=True, schema=schema)

# COMMAND ----------

# DBTITLE 1,Load and clean MassCEC data
# Load MassCEC data
mass_data = pd.read_excel(
    str(G.RAW_DATA_FPATH / "gshp" / "ResidentialandSmallScaleGSHPProjectDatabase.xlsx"), skiprows=2
)
# Transform schema
mass_data = mass_data.rename(
    columns={
        "Site City/Town": "city",
        "Building Size (Square Feet)": "building sq ft",
        "Previous Heat Distribution": "previous heat distribution",
        "Heat Pump Function": "hp function",
        "Total Bore Hole Depth (ft)": "total bore depth feet",
        "Total System Cost": "total system cost",
        "Number of Bore Holes": "number of bores",
    }
)
mass_data["total bore depth feet"] = pd.to_numeric(
    mass_data["total bore depth feet"].apply(lambda x: None if x == "-" else x)
)
mass_data["number of bores"] = pd.to_numeric(mass_data["number of bores"].apply(lambda x: None if x == "-" else x))
mass_data["total system cost"] = pd.to_numeric(mass_data["total system cost"].apply(lambda x: str(x).replace("$", "")))
mass_data["total capacity tons"] = mass_data["Total Capacity (BTU/hr)"] / BTU_PER_TON
mass_data["database"] = "MassCEC"
mass_data["state"] = "MA"

# Before union-ing with NYSERDA data, apply some filters:
# Remove rows with missing sq ft
mass_data = mass_data.dropna(subset="building sq ft", axis=0)
mass_data["building sq ft"] = mass_data["building sq ft"].apply(lambda x: float(str(x).replace(",", "")))
# Remove water to water GSHPs
mass_data = mass_data[mass_data["Pump type"].apply(lambda x: x.lower()) != "water to water"]
# Keep only closed loop vertical systems
mass_data = mass_data[mass_data["Loop Config."] == "Closed loop - vertical"]

# Select columns of interest
mass_data = mass_data[
    [
        "city",
        "state",
        "building sq ft",
        "previous heat distribution",
        "hp function",
        "total bore depth feet",
        "total capacity tons",
        "total system cost",
        "database",
        "number of bores",
    ]
]

# COMMAND ----------

# DBTITLE 1,Combine NYSERDA and MassCEC data
df = ny_data.union(spark.createDataFrame(mass_data, schema=schema))
df.count()

# COMMAND ----------

# MAGIC %md ### Clean Columns and Add Geographic Values

# COMMAND ----------

# DBTITLE 1,Clean column names
df_clean = dcutils.clean_colnames(df)

# COMMAND ----------

print(f"Number of rows before filtering out N/A values in features: {df_clean.count()}")

# COMMAND ----------

# DBTITLE 1,Drop Rows with Null Values in Specific Columns
df_clean_revised = df_clean.dropna(subset=["building_sq_ft", "total_capacity_tons"]).withColumn(
    "city_state", F.concat(F.col("city"), F.lit(", "), F.col("state")).cast("string")
)

# COMMAND ----------

print(f"Number of rows after filtering out N/A values in features: {df_clean_revised.count()}")

# COMMAND ----------

# DBTITLE 1,County geoid extraction and join
# Because geocoding is slow, we load geocoded county values from a table.
# If specified, overwrite table by using the Google geocoding API to extract a lat lon and the census API to find the associated county geoid.

# Note: we're considering this table a raw input file and thus do not control its version.
geocoded_table_path = "processed.upfront_costs.gshp_geocoded_city_states"
if spark.catalog.tableExists(geocoded_table_path) and not RE_GEOCODE:
    city_state_county = spark.table(geocoded_table_path)
else:
    # Extract the county geoid using geocoding
    GEOCODER = geocoding_util.CountyGeocoder(GOOGLE_API_KEY)
    city_state_county = geocoding_util.process_pyspark_dataframe(df_clean_revised, "city_state", GEOCODER)
    city_state_county = spark.createDataFrame(city_state_county)
    city_state_county.write.saveAsTable(
        geocoded_table_path,
        format="delta",
        mode="overwrite",
        overwriteSchema=True,
    )

# Join the county geoids back to the mass data based on the city_state column
df_county_geoid = df_clean_revised.join(city_state_county, on="city_state", how="left")

# COMMAND ----------

# MAGIC %md ### Add in RSMeans Values for Scaling

# COMMAND ----------

# DBTITLE 1,Join RSMeans data
df_rs_means = (
    spark.read.table("upfront_cost.rsmeans_county")
    .where(F.col("category") == "fire suppression, plumbing & hvac")
    .where(F.col("component") == "total")
)

df_join = df_county_geoid.join(df_rs_means, on="county_geoid", how="left")

# COMMAND ----------

# Scale by RSMeans to get adjusted system costs
df_join = df_join.withColumn("adjusted_system_cost", F.col("total_system_cost") / (F.col("rsmeans_price_index") / 100))

# COMMAND ----------

# MAGIC %md ### Write Out Data

# COMMAND ----------

# DBTITLE 1,Write out as delta table
sparkdb.write_delta_table(
    df=df_join,
    db_table_name="processed.upfront_costs.gshp_training_features",
    mode=MODE,
    release_version=RELEASE_VERSION,
)
