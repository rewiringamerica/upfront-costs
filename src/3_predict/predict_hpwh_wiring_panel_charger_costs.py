# Databricks notebook source
# MAGIC  %md
# MAGIC # Upfront Costs for HPWH, Wiring, Electric Panels, and EV Chargers
# MAGIC ### Goal
# MAGIC To predict the upfront costs of panel upgrades, wiring, heat pump water heaters, and EV chargers.
# MAGIC
# MAGIC ### Process
# MAGIC - Panels: the estimated cost of a panel upgrade varies by the target service amperage. The absolute
# MAGIC level is anchored to the median main-service-panel upgrade cost in this [2024 presentation](https://homes.lbl.gov/sites/default/files/2024-04/BPA_NHPC24-Affordable%20Electrification_final.pdf) by LBNL & NREL,
# MAGIC and scaled across amperage tiers by a relative cost multiplier derived from RSMeans residential electric
# MAGIC service assemblies and cross-checked against industry cost guides (see `PANEL_AMPERAGE_COST_MULTIPLIERS`).
# MAGIC - Wiring: take the difference in cost between projects replacing an electric resistance water
# MAGIC heater and projects replacing a gas water heater in the TECH Clean CA data.
# MAGIC - Heat pump water heaters: take the median price of a heat pump water heater install that is not
# MAGIC replacing electric resistance, then subtract the wiring cost.
# MAGIC - EV L2 charger: the estimated cost of an L2 charger is drawn from 2025 cost projections in the [2019 ICCT EV Charging Cost study](https://theicct.org/wp-content/uploads/2021/06/ICCT_EV_Charging_Cost_20190813.pdf).
# MAGIC
# MAGIC ### I/O
# MAGIC ##### Inputs:
# MAGIC - `processed.upfront_costs.ashp_hpwh_training_features`
# MAGIC - BuildingSet metadata
# MAGIC - `upfront_cost.rsmeans_county`
# MAGIC # MAGIC
# MAGIC ##### Outputs:
# MAGIC - `processed.<building_set>.hpwh_wiring_panel_charger_cost_by_building`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

from functools import reduce

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from dmlutils import sparkdb
from dmlutils.building_upgrades.upgrades import (
    BuildingSet,
    get_building_metadata_table,
    get_table_name_for_building_set,
    national_building_metadata,
)

from src import util

# COMMAND ----------

# Set I/O mode
dbutils.widgets.dropdown(
    "mode", "feature", ["feature", "main"]
)  # mode to read/write delta tables in (see `sparkdb.modify_db_table_name()`)
MODE = dbutils.widgets.get("mode")

# Choose BuildingSet
dbutils.widgets.dropdown(
    "building_set", choices=[b.value for b in BuildingSet], defaultValue=BuildingSet.RESSTOCK_2024_2
)
BUILDING_SET = dbutils.widgets.get("building_set")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Establish Constants

# COMMAND ----------

# percentiles take in aggregation
PERCENTILE_VALUES = [0.20, 0.25, 0.5, 0.75, 0.80]

# Load current release version so that we can load the proper model number
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md ### Define Functions
# MAGIC

# COMMAND ----------


def get_percentiles(df, groupby_cols, percentile_values=[0.25, 0.5, 0.75]):
    """
    Get the percentiles of "project_costs" grouped over `groupby_cols`.

    With one column for each of the percentiles p specified in percentile_values,
    named as "upfront_cost_{int(p*100)}percentile", unless p = .5 in which case it is
    named "upfront_cost_median".

    Args:
        df (pyspark.sql.dataframe.DataFrame): a PySpark dataframe with a project_cost column

    Returns
    -------
        df (pyspark.sql.dataframe.DataFrame): a PySpark dataframe with one column for each
            groupby col and for each percentile
    """
    # get the percentile estimates
    df_grouped = df.groupby(groupby_cols).agg(
        F.percentile_approx("project_cost_adjusted", percentile_values).alias("cost_percentiles")
    )
    # pull out one column for each percentile
    for i, p in enumerate(percentile_values):
        colname = "upfront_cost_median" if p == 0.5 else f"upfront_cost_{int(p*100)}percentile"
        df_grouped = df_grouped.withColumn(colname, df_grouped.cost_percentiles[i])

    return df_grouped.drop("cost_percentiles")


# COMMAND ----------

# MAGIC %md ### Load data

# COMMAND ----------

# HPWH cost data
upfront_cost_training_features = spark.table(
    sparkdb.modify_db_table_name("processed.upfront_costs.ashp_hpwh_training_features", MODE)
)

# BuildingSet Metadata
building_metadata_table_path = get_building_metadata_table(BUILDING_SET)
building_set_metadata = spark.table(building_metadata_table_path)

# RS Means data
rsmeans = spark.table("upfront_cost.rsmeans_county")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Panel Upgrades

# COMMAND ----------

# Panel upgrade cost by target service amperage.
# Cost = LBNL absolute level x RSMeans-derived amperage multiplier.
#
# Level: LBNL/NREL NHPC 2024 deck reports a median main-service-panel upgrade of $3,499 (n=167), range
# $1k-$5k (2023 USD, national). Pinned to the 200A tier.
# https://homes.lbl.gov/sites/default/files/2024-04/BPA_NHPC24-Affordable%20Electrification_final.pdf

PANEL_200A_MEDIAN_COST = 3500

# Multipliers are relative to 200A, from RSMeans 091000 Electric Service (Residential Square Foot Assemblies,
# national average), which prices a full residential service by amperage. Only the ratios matter here -- the
# level is LBNL's, and the RSMeans county price index localizes costs downstream, so these stay national.
#   RSMeans Total: 100A $1,093 / 200A $1,818 / 400A $10,329  ->  ratios 0.60 / 1.00 / 5.68
#
# Cross-checked against industry cost guides: 100A ~0.6, 300A ~1.75, 400A ~2.7-5.0.
#   https://homeguide.com/costs/cost-to-replace-electrical-panel
#   https://home.costhelper.com/electrical-upgrading-electrical-service.html
#   https://www.angi.com/articles/ask-angie-what-does-it-cost-upgrade-200-amps.htm
#   https://www.fixr.com/costs/install-electrical-circuit-panel-upgrade
#
# Per-amperage ratio choices:
#   150A 0.80 - interpolated 100A-200A; both sources agree 100A is 0.60.
#   200A 1.00 - anchor (LBNL median).
#   300A 2.00 - no RSMeans 300A multiplier. A linear 200A-400A interp gives 3.34 but overshoots, since the cost curve is
#               convex (the jump is the 400A service, not 300A). Industry guides say ~1.75, so we use 2.0.
#   400A 4.00 - RSMeans says 5.68, guides span 2.7-5.0 (simple swap vs full service upgrade); 4.0 is a compromise.
#               Rare in practice -- pge_analysis only recommends 400A when planned load exceeds 300A.
#
# RSMeans figures are from the 2018 release; worth refreshing eventually, though ratios are stable across editions.

PANEL_AMPERAGE_COST_MULTIPLIERS = {
    150: 0.80,
    200: 1.00,  # LBNL anchor
    300: 2.00,
    400: 4.00,
}

# 200A percentile costs from the LBNL distribution ($1,000/$1,500/$4,500/$5,000 against the $3,500 median).
# Scaled by each tier's multiplier below.
PANEL_COST_PERCENTILES = {
    "upfront_cost_20percentile": 1000,
    "upfront_cost_25percentile": 1500,
    "upfront_cost_75percentile": 4500,
    "upfront_cost_80percentile": 5000,
}

# Build one cost row per amperage tier.
panel_cost_rows = [
    (
        f"Panel Upgrade {amperage}A",
        float(round(PANEL_200A_MEDIAN_COST * multiplier)),
        *[float(round(cost * multiplier)) for cost in PANEL_COST_PERCENTILES.values()],
    )
    for amperage, multiplier in PANEL_AMPERAGE_COST_MULTIPLIERS.items()
]

panel_upfront_cost = spark.createDataFrame(
    panel_cost_rows,
    [
        "appliance_option",
        "upfront_cost_median",
        *PANEL_COST_PERCENTILES.keys(),
    ],
)

# Cross join gives one row per (building, amperage tier), each a distinct "Panel Upgrade {amps}A" option.
buildings_with_panel_costs = building_set_metadata.select("building_id", "county_geoid").join(
    panel_upfront_cost, how="cross"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### EV L2 Charger
# MAGIC From the 2019 ICCT EV Charging Cost study, which use a 3% per year reduction in cost to estimate costs for a home L2 charger installation in 2025.
# MAGIC Following the report methodology, we estimate the cost of an attached home installation as the average of the costs of a detached and apartment installation.

# COMMAND ----------

APARTMENT_L2_CHARGER_COST = 3933.0
DETACHED_L2_CHARGER_COST = 1317.0
ATTACHED_L2_CHARGER_COST = (APARTMENT_L2_CHARGER_COST + DETACHED_L2_CHARGER_COST) / 2

# Create table of hard-coded costs
l2_upfront_cost = spark.createDataFrame(
    [
        (
            "L2 EV Charger",
            "Single-Family Detached",
            DETACHED_L2_CHARGER_COST,
            # Assume a long-tailed distribution of costs, with a 20th percentile = 70% of the median and a 80th percentile = 300% of the median
            DETACHED_L2_CHARGER_COST * 0.7,
            DETACHED_L2_CHARGER_COST * 0.75,
            DETACHED_L2_CHARGER_COST * 2.5,
            DETACHED_L2_CHARGER_COST * 3,
        ),
        (
            "L2 EV Charger",
            "Single-Family Attached",
            ATTACHED_L2_CHARGER_COST,
            # Assume a long-tailed distribution of costs, with a 20th percentile = 70% of the median and a 80th percentile = 300% of the median
            ATTACHED_L2_CHARGER_COST * 0.7,
            ATTACHED_L2_CHARGER_COST * 0.75,
            ATTACHED_L2_CHARGER_COST * 2.5,
            ATTACHED_L2_CHARGER_COST * 3,
        ),
    ],
    schema=StructType(
        [
            StructField("appliance_option", StringType(), True),
            StructField("in_geometry_building_type_acs", StringType(), True),
            StructField("upfront_cost_median", FloatType(), True),
            StructField("upfront_cost_20percentile", FloatType(), True),
            StructField("upfront_cost_25percentile", FloatType(), True),
            StructField("upfront_cost_75percentile", FloatType(), True),
            StructField("upfront_cost_80percentile", FloatType(), True),
        ]
    ),
)

# Round costs for cleanliness
l2_upfront_cost = (
    l2_upfront_cost.withColumn("upfront_cost_median", F.round(F.col("upfront_cost_median")))
    .withColumn("upfront_cost_20percentile", F.round(F.col("upfront_cost_20percentile")))
    .withColumn("upfront_cost_25percentile", F.round(F.col("upfront_cost_25percentile")))
    .withColumn("upfront_cost_75percentile", F.round(F.col("upfront_cost_75percentile")))
    .withColumn("upfront_cost_80percentile", F.round(F.col("upfront_cost_80percentile")))
)

l2_upfront_cost.display()

# COMMAND ----------

# Join national level panel cost estimate against each building ID
buildings_with_l2_costs = (
    building_set_metadata.select("building_id", "in_geometry_building_type_acs", "county_geoid")
    .join(l2_upfront_cost, on=["in_geometry_building_type_acs"], how="left")
    .drop("in_geometry_building_type_acs")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wiring
# MAGIC From TECH Clean CA data. Comparing the cost of a heat pump water install when replacing electric resistance
# MAGIC (no new circuit required) to replacing natural gas (new circuit required).
# MAGIC

# COMMAND ----------

# display the number of ER samples used to create one half of
# the comparison used to create this estimate
print(
    f"Number of samples: {upfront_cost_training_features.where((col('product_type') == 'Heat Pump Water Heater') & (col('previous_wh_fuel_type') == 'Electric Resistance')).count()}"
)

# COMMAND ----------

# display the number of natural gas samples used to create one half of
# the comparison used to create this estimate
print(
    f"Number of samples: {upfront_cost_training_features.where((col('product_type') == 'Heat Pump Water Heater') & (col('previous_wh_fuel_type') == 'Natural Gas')).count()}"
)

# COMMAND ----------

# get the quartile estimates of cost for heat pump water heater costs replacing gas vs. replacing electric resistance
wiring_cost_data = get_percentiles(
    df=(
        upfront_cost_training_features.where((col("product_type") == "Heat Pump Water Heater")).where(
            col("previous_wh_fuel_type").isin("Natural Gas", "Electric Resistance")
        )
    ),
    groupby_cols=["previous_wh_fuel_type"],
    percentile_values=PERCENTILE_VALUES,
)

# switch to Pandas to transpose and take the difference between the gas and electric resistance costs
wiring_cost_data_pd = wiring_cost_data.toPandas()
wiring_cost_data_pd.sort_values(by="previous_wh_fuel_type", inplace=True)
wiring_cost_data_pd.dropna(inplace=True)
wiring_cost_data_pd = wiring_cost_data_pd.select_dtypes(include=["number"]).diff().dropna(axis=0)

# pass back to Spark to create final dataframe
wiring_upfront_cost = spark.createDataFrame(wiring_cost_data_pd).withColumn("appliance_option", F.lit("240V Circuit"))

# COMMAND ----------

# Join national level wiring cost estimate against each building ID
buildings_with_wiring_costs = building_set_metadata.select("building_id", "county_geoid").join(
    wiring_upfront_cost, how="cross"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Heat Pump Water Heaters
# MAGIC From Tech Clean CA data. Taking the median cost of a heat pump water install that is not replacing electric
# MAGIC resistance, than subtracting the wiring cost from above. We do this instead of just looking at the cost of
# MAGIC electric resistance replacement water heater installations because there are so many fewer of those than there
# MAGIC are fossil fuel replacement, so we would lose a lot of our sample size by looking only at electric resistance
# MAGIC replacement.
# MAGIC

# COMMAND ----------

# display the number of samples used to create this cost estimate
print(
    f"Number of samples: {upfront_cost_training_features.where((col('product_type') == 'Heat Pump Water Heater') & (col('previous_wh_fuel_type') != 'Electric Resistance')).count()}"
)

# get the quartile estimates of cost for heat pump water heater installations that are not replacing electric resistance
hpwh_upfront_cost_with_wiring = get_percentiles(
    df=upfront_cost_training_features.where(
        (col("product_type") == "Heat Pump Water Heater") & (col("previous_wh_fuel_type") != "Electric Resistance")
    ),
    groupby_cols=[],
    percentile_values=PERCENTILE_VALUES,
)

# COMMAND ----------

# Subtract median wiring upfront cost from HPWH cost estimates to get cost estimates without wiring
wiring_cost_median = wiring_upfront_cost.collect()[0]["upfront_cost_median"]

hpwh_upfront_cost = hpwh_upfront_cost_with_wiring.select(
    [(col(c) - wiring_cost_median).alias(c) for c in hpwh_upfront_cost_with_wiring.columns]
)

# COMMAND ----------

# Join national level HPHW cost estimate against each building ID
buildings_with_hpwh_costs = (
    building_set_metadata.select("building_id", "county_geoid")
    .join(hpwh_upfront_cost, how="cross")
    .withColumn("appliance_option", F.lit("Heat Pump Water Heater"))
)

# COMMAND ----------

# MAGIC %md ## Combine Upfront Costs from Different Upgrades

# COMMAND ----------

# Call unionByName on the instance (not the unbound DataFrame.unionByName), which reaches into _jdf and is
# blocked on shared-access Unity Catalog clusters.
location_varying_upfront_cost = reduce(
    lambda left, right: left.unionByName(right),
    [buildings_with_panel_costs, buildings_with_l2_costs, buildings_with_wiring_costs, buildings_with_hpwh_costs],
)

# COMMAND ----------

# reorder columns to put appliance_option first
# Percentile columns listed explicitly rather than via colRegex, which reaches into _jdf and is blocked on
# shared-access Unity Catalog clusters.
location_varying_upfront_cost_by_building = location_varying_upfront_cost.select(
    "building_id",
    "county_geoid",
    "appliance_option",
    "upfront_cost_median",
    "upfront_cost_20percentile",
    "upfront_cost_25percentile",
    "upfront_cost_75percentile",
    "upfront_cost_80percentile",
)

# COMMAND ----------

# Scale national cost estimates for each of the three upgrades by the rsmeans index for each buildings location to
# localize costs
adjusted_location_varying_costs = (
    location_varying_upfront_cost_by_building.join(
        rsmeans.where(F.col("component") == "total").select("county_geoid", "rsmeans_price_index", "category"),
        on="county_geoid",
    )
    .where(
        # Use different RSMeans category for wiring/panel upgrade and HPWH
        (
            (
                (F.col("category") == "electrical, communications & util.")
                & (
                    F.col("appliance_option").startswith("Panel Upgrade")
                    | F.col("appliance_option").isin(["240V Circuit", "L2 EV Charger"])
                )
            )
            | (
                (F.col("category") == "fire suppression, plumbing & hvac")
                & (F.col("appliance_option") == "Heat Pump Water Heater")
            )
        )
    )
    .drop("category")
    .withColumn("upfront_cost_median", F.col("upfront_cost_median") * F.col("rsmeans_price_index") / 100)
    .withColumn("upfront_cost_20percentile", F.col("upfront_cost_20percentile") * F.col("rsmeans_price_index") / 100)
    .withColumn("upfront_cost_25percentile", F.col("upfront_cost_25percentile") * F.col("rsmeans_price_index") / 100)
    .withColumn("upfront_cost_75percentile", F.col("upfront_cost_75percentile") * F.col("rsmeans_price_index") / 100)
    .withColumn("upfront_cost_80percentile", F.col("upfront_cost_80percentile") * F.col("rsmeans_price_index") / 100)
)

# COMMAND ----------

# MAGIC %md ## Write out table

# COMMAND ----------

# Specify BuildingSet in a table column and use selective overwrite for national BuildingSets.
adjusted_location_varying_costs = adjusted_location_varying_costs.withColumn("building_set", F.lit(BUILDING_SET))
write_options = {}
if BUILDING_SET in national_building_metadata:
    write_options = {"overwriteSchema": "false", "replaceWhere": f"building_set = '{BUILDING_SET}'"}

# Determine output table name based on building set
db_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="processed",
    schema="upfront_costs",
    table_name="hpwh_wiring_panel_charger_cost_by_building",
)

# Write out table
sparkdb.write_delta_table(
    df=adjusted_location_varying_costs,
    db_table_name=db_table_name,
    mode=MODE,
    release_version=RELEASE_VERSION,
    **write_options,
)
