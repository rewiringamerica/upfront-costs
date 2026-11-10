# Databricks notebook source
# MAGIC  %md
# MAGIC
# MAGIC # Process Upfront Costs data for EVs
# MAGIC
# MAGIC ### Goal
# MAGIC Produce a table of EV upfront costs by vehicle category and range.
# MAGIC
# MAGIC ### Process
# MAGIC - Scrape data from the PG&E EV comparison tool (ev.pge.com), which uses underlying data from ZappyRide, a JD Power product.
# MAGIC - Process scraped data to produce a table of EV upfront costs by vehicle category and range
# MAGIC Note that the vehicle categories and range bins are aligned with those used in the ResStock modeling of EVs, so that we can join on these features when we join to building metadata in the next step.
# MAGIC Note also that this notebook is not part of the upfront-costs transmission line job because it should only be run manually when we want to pull down new EV cost estimates.
# MAGIC
# MAGIC ### I/O
# MAGIC ##### Inputs:
# MAGIC - BuildingSet metadata
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `processed.<building_set>.ev_upfront_cost_by_building`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

# imports
import pandas as pd
import requests

from dmlutils import sparkdb

from src import util

# COMMAND ----------

dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

# Load current release version so that we can load the proper model number
RELEASE_VERSION = util.get_release_version()

# We need a ZappyRide API key to query EV cost data. It's a free service, but we still
# want to hide the API key as best practice.
# To generate a new key, copy the X-Api-Key header from a request to app.production.zappyride.com in the
# network tab of your browser's dev tools while loading https://ev.pge.com/compare.
ZAPPYRIDE_API_KEY = dbutils.secrets.get(scope="datascience", key="zappyride_api_key")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scrape EV data from PG&E website
# MAGIC The [PG&E EV comparison tool](ev.pge.com) uses underlying data from [ZappyRide](https://www.jdpower.com/business/zappyride), a JD Power product.

# COMMAND ----------

# Headers copied from web browser
headers = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://ev.pge.com",
    "Referer": "https://ev.pge.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "X-Api-Key": ZAPPYRIDE_API_KEY,
    "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

# Params also copied from web browser and pared down to only what appears to be strictly necessary.
# Note that the API requires either a postal code or a lat/lon. PG&E's website uses postal code 94108.
# Experimentation with multiple postal codes showed that changing the value had no effect on the cost results,
# so we keep 94108.
params = {"postcode": "94108"}

# Run request
response = requests.get("https://app.production.zappyride.com/vehicles", params=params, headers=headers, timeout=30)
response.raise_for_status()

# COMMAND ----------

# Postprocess the response
vehicles = pd.DataFrame(response.json()["vehicles"])
# Bucket range into 200 and 300 buckets
vehicles["range"] = vehicles["total_range"].map(
    lambda x: 200 if (x < 300 and x >= 200) else 300 if ((x >= 300) and (x < 400)) else None
)

# Filter to only the set of vehicles we want:
filtered_vehicles = vehicles[
    # Fully-electric EVs
    (vehicles["fuel"] == "BEV")
    &
    # Vehicles with range within one of the two buckets
    (vehicles["range"].notna())
    &
    # Filter out used vehicles
    (vehicles["available_for_purchase_as_new"])
    &
    # Filter out vehicles with MSRP > $125,000, which are usually luxury cars
    (vehicles["msrp"] < 125000)
].copy()

# Make sure electric range = total range for all rows
assert (filtered_vehicles["total_range"] == filtered_vehicles["electric_range"]).all()
filtered_vehicles["range"] = filtered_vehicles["range"].astype(int)

# Map the Zappy form factor to our ResStock vehicle categories
filtered_vehicles["vehicle_category"] = filtered_vehicles["form_factor"].map(
    {
        "pickup_truck": "Pickup",
        "SUV": "SUV",
        "Crossover": "SUV",
        "Sedan": "Midsize",
        "Minivan": "Midsize",
        "Station Wagon": "Midsize",
        "Coupe": "Compact",
        "Hatchback": "Compact",
    }
)
# Make sure all `form_factor` values were mapped
unmapped = filtered_vehicles[filtered_vehicles["vehicle_category"].isna()]["form_factor"].unique()
assert len(unmapped) == 0, f"Unmapped form_factors: {unmapped}"

# Calculate MSRP statistics by vehicle category and range
ev_grouped_costs = filtered_vehicles.groupby(["vehicle_category", "range"], as_index=False).agg(
    upfront_cost_median=pd.NamedAgg(column="msrp", aggfunc="median"),
    upfront_cost_20percentile=pd.NamedAgg(column="msrp", aggfunc=lambda x: x.quantile(0.2)),
    upfront_cost_80percentile=pd.NamedAgg(column="msrp", aggfunc=lambda x: x.quantile(0.8)),
    n_vehicles=pd.NamedAgg(column="msrp", aggfunc="count"),
)
# Convert cost columns to int for cleanliness
for column in ["upfront_cost_median", "upfront_cost_20percentile", "upfront_cost_80percentile"]:
    ev_grouped_costs[column] = ev_grouped_costs[column].round(0).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert to spark and save to catalog

# COMMAND ----------

# Convert to Spark DataFrame
ev_grouped_costs_spark = spark.createDataFrame(ev_grouped_costs)

# Save to catalog for downstream use
sparkdb.write_delta_table(
    df=ev_grouped_costs_spark,
    db_table_name="processed.upfront_costs.ev_upfront_cost_by_vehicle_category_range",
    mode=MODE,
    release_version=RELEASE_VERSION,
)
