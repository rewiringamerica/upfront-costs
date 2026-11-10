# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Process Mass Whole Home Pricing Data
# MAGIC
# MAGIC ### Goal
# MAGIC Process pricing data Mass Whole Home Pilot 2019-2021 for use in upfront cost model training.
# MAGIC
# MAGIC ### Process
# MAGIC - Standardize and clean column names
# MAGIC - Extract SEER/HSPF efficiencies from AHRI data and join to Mass heat pump data based on model number
# MAGIC - Create hash id
# MAGIC - Write to delta table
# MAGIC
# MAGIC
# MAGIC ### Inputs
# MAGIC - `raw / pricing / mass_wholehome_data.csv`
# MAGIC - `raw / pricing / wholehome_efficiency.csv`
# MAGIC - `processed.upfront_costs.mass_wholehome_geocoded_city_states` (Optional)
# MAGIC
# MAGIC ### Outputs
# MAGIC
# MAGIC - `processed.upfront_costs.mass_wholehome`
# MAGIC
# MAGIC ### To Do
# MAGIC - Eventually, work with Tom C. on an AHRI API to get efficiencies for all specified models

# COMMAND ----------

# MAGIC %md ### Set Up

# COMMAND ----------

# imports
from itertools import chain

import pandas as pd
import pyspark.sql.functions as F

# internal
from dmlutils import constants as C
from dmlutils import data_cleaning_utils as dcutil
from dmlutils import hashutils, sparkdb

import src.geocoding_util as geocoding_util
import src.globals as G
import src.util as util

# COMMAND ----------

dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
dbutils.widgets.dropdown("re_geocode", "False", ["True", "False"])  # whether to re-geocode the records
MODE = dbutils.widgets.get("mode")
RE_GEOCODE = eval(dbutils.widgets.get("re_geocode"))

# COMMAND ----------

# Load current release version to version feature table written at end of script
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# file paths
PRICING_RAW_DATA_FPATH = G.RAW_DATA_FPATH / "pricing"

# Establish API Key
GOOGLE_API_KEY = dbutils.secrets.get(scope="datascience", key="google_geocoding_api_key")

# Initialize the geocoder
GEOCODER = geocoding_util.CountyGeocoder(GOOGLE_API_KEY)

# COMMAND ----------

# MAGIC %md ### Load Data

# COMMAND ----------

# Mass whole home
path_mass_wholehome = PRICING_RAW_DATA_FPATH / "mass_wholehome_data.csv"
mass_wholehome = pd.read_csv(
    path_mass_wholehome,
    converters={"Total Estimated Installed Project Cost": lambda s: s.replace("$", "").replace(",", "")},
    thousands=",",
    decimal=".",
    dtype="str",
    na_values=[" Not Applicable "],
)

# clean up column name to have no annoying punctuation and be all lower case
mass_wholehome = dcutil.clean_colnames(spark.createDataFrame(mass_wholehome))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filling null HSPF and SEER values
# MAGIC We downloaded csvs from AHRI to find SEER/HSPF efficiencies based on the heat pump model number.
# MAGIC After joining based on the model number, there are still null efficiencies. For these, we
# MAGIC take the mean across all SEER/HSPF values and join these.

# COMMAND ----------

# load the efficiency csv
wholehome_efficiency = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .csv(str(PRICING_RAW_DATA_FPATH / "wholehome_efficiency.csv"))
    .select("ahri_number_of_outdoor_unit_number1", "seer", "hspf")
)

# join with mass whole home dataframe on the AHRI number
mass_wholehome_efficiencies = mass_wholehome.join(
    wholehome_efficiency, "ahri_number_of_outdoor_unit_number1", how="left"
)

# calculate HSPF and SEER means for when there are null values in the joined table
stats_wholehome = mass_wholehome_efficiencies.select(
    F.mean(F.col("hspf")).alias("hspf_mean"),
    F.mean(F.col("seer")).alias("seer_mean"),
).collect()

hspf_mean_wholehome, seer_mean_wholehome = stats_wholehome[0]

# COMMAND ----------

# Dictionary to replace city names
city_dict = {
    "Watertown": "Watertown Town",
    "Bedford": "New Bedford",
    "Methuen": "Methuen Town",
    "Winthrop": "Winthrop Town",
    "Mashpee": "Mashpee Neck",
    "Mattapoisett": "Mattapoisett Center",
    "Amesbury": "Amesbury Town",
    "West Newton": "Newton",
    "Northamtpon": "Northampton",
    "Amherst": "Amherst Town",
    "East Orleans": "Orleans",
    "North Andover": "Andover",
    "South Boston": "Boston",
}

# Create mapping expression
mapping_expr = F.create_map([F.lit(x) for x in chain(*city_dict.items())])

# standardize some columms for upfront cost dataframe
cost_mass_wholehome_raw = (
    mass_wholehome_efficiencies.withColumnRenamed("rebate_approved_by_masscec", "start_date")
    .withColumn("year", F.concat(F.lit("20"), F.substring(F.col("start_date"), -2, 2)))
    .withColumn("month", F.split(F.col("start_date"), "/").getItem(0))
    .withColumn("month", F.lpad(F.col("month"), 2, "0"))
    .withColumn(
        "square_footage",
        F.col("total_conditioned_square_footage_of_the_home_sq_ft").cast("double"),
    )
    .withColumn(
        "project_cost_raw",
        F.col("total_estimated_installed_project_cost").cast("double"),
    )
    .withColumn("heat_pump_type", F.lit("Whole Home"))
    .withColumn("state_geoid", F.lit("25"))
    .withColumn("program", F.lit("Mass Whole Home"))
    .withColumn("is_cold_climate", F.lit(True))
    .withColumn("product_type", F.lit("Heat Pump HVAC"))
    .withColumn(
        "heating_capacity_tons",
        F.col("total_capacity_of_outdoor_units_at_design_temperature_btu_hr") / C.BTU_PER_TON,
    )
    .withColumn(
        "had_duct_replacement",
        F.when(
            F.col("estimated_cost_of_ductwork_costs_if_applicable") == "$0.00",
            F.lit(False),
        ).otherwise(F.lit(True)),
    )
    .withColumn(
        "had_panel_upgrade",
        F.when(F.col("estimated_cost_of_electrician_costs") == "$0.00", F.lit(False)).otherwise(F.lit(True)),
    )
    # set HSPF and SEER to the mean when they are null
    .withColumn(
        "hspf_region_iv",
        F.when(F.col("hspf").isNotNull(), F.col("hspf")).otherwise(hspf_mean_wholehome),
    )
    .withColumn(
        "seer",
        F.when(F.col("seer").isNotNull(), F.col("seer")).otherwise(seer_mean_wholehome),
    )
    .withColumn("ducted", F.when(F.col("outdoor_unit_type") == "Centrally Ducted", F.lit(True)))
    .withColumn(
        "city",
        F.when(F.col("city").isin(list(city_dict.keys())), mapping_expr[F.col("city")]).otherwise(F.col("city")),
    )
    .withColumn("city", F.initcap(F.col("city")))
    .withColumn("city_state", F.concat(F.col("city"), F.lit(", "), F.lit("MA")))
)

# There are several records that are exactly the same across all columns, we can check with this code:
# cost_mass_wholehome_raw.groupby('id').count().where(F.col('count')>1).display()
# We need to remove these duplicates
cost_mass_wholehome_raw = cost_mass_wholehome_raw.dropDuplicates(cost_mass_wholehome_raw.columns)

# Add a hash id to uniquely identify records
cost_mass_wholehome_raw = hashutils.add_id_hash_col(cost_mass_wholehome_raw, cost_mass_wholehome_raw.columns, True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adding counties
# MAGIC Because geocoding is slow, we load geocoded county values from a table.
# MAGIC If specified, overwrite table by using the Google geocoding API to extract a lat lon and the census API to find the associated county geoid.

# COMMAND ----------

# Note: we're considering this table a raw input file and thus do not control its version.
geocoded_table_path = "processed.upfront_costs.mass_wholehome_geocoded_city_states"
if spark.catalog.tableExists(geocoded_table_path) and not RE_GEOCODE:
    city_state_county = spark.table(geocoded_table_path)
else:
    # Extract the county geoid using geocoding
    city_state_county = geocoding_util.process_pyspark_dataframe(cost_mass_wholehome_raw, "city_state", GEOCODER)
    city_state_county = spark.createDataFrame(city_state_county)
    city_state_county.write.saveAsTable(
        geocoded_table_path,
        format="delta",
        mode="overwrite",
        overwriteSchema=True,
    )

# Join the county geoids back to the mass data based on the city_state column
cost_mass_wholehome_county_geoid = cost_mass_wholehome_raw.join(city_state_county, on="city_state", how="left")

# COMMAND ----------

# DBTITLE 1,Write table
sparkdb.write_delta_table(
    df=cost_mass_wholehome_county_geoid,
    db_table_name="processed.upfront_costs.mass_wholehome",
    mode=MODE,
    release_version=RELEASE_VERSION,
)
