# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Weatherization Upfront Costs
# MAGIC
# MAGIC ### Goal
# MAGIC To produce a table with predicted upfront costs of various weatherization packages in ResStock (e.g., Basic, Enhanced, and Light Touch) at the building level.
# MAGIC We assume these costs vary by location and by square footage. Each package contains one or more weatherization measures.
# MAGIC
# MAGIC ### Process
# MAGIC
# MAGIC Costs are based on LBNL study ["The Cost of Decarbonization and Energy Upgrade Retrofit
# MAGIC for US Homes](https://escholarship.org/content/qt0818n68p/qt0818n68p.pdf). In particular,
# MAGIC we use cost estimates from LBNL's summaries of their deep energy retrofit (DER) database. Note that this database dates
# MAGIC back to 2005-2010, with costs adjusted for inflation.
# MAGIC
# MAGIC - Construct table of median cost/sqft estimates by measure from a LBNL cost study
# MAGIC - Aggregate measures comprising Basic and Enhanced packages (Light Touch package contains only one measure)
# MAGIC - Adjust to present-day costs using construction inflation indices
# MAGIC - Adjust to location-specific costs using RSMeans
# MAGIC
# MAGIC ### I/O
# MAGIC ##### Inputs:
# MAGIC - `upfront_cost.construction_inflation`: used to adjust for inflation
# MAGIC - `upfront_cost.rsmeans_county`: used to adjust for location
# MAGIC - BuildingSet metadata: use to pull sqftage bins
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `processed.<building_set>.weatherization_cost_by_building_package`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

from dmlutils import sparkdb
from dmlutils.building_upgrades.upgrades import (
    BuildingSet,
    Weatherization,
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

# Load current release version
RELEASE_VERSION = util.get_release_version()

# COMMAND ----------

# MAGIC %md ### Read in / Construct Data

# COMMAND ----------

# BuildingSet metadata
building_metadata_table_path = get_building_metadata_table(BUILDING_SET)
building_set_metadata = spark.table(building_metadata_table_path)

# Get full set of counties
resstock_counties = building_set_metadata.select("county_geoid").distinct()

# Inflation adjustment factors
construction_inflation = spark.table(sparkdb.modify_db_table_name("upfront_cost.construction_inflation", MODE))

# Location adjustment factors
rsmeans = spark.table("upfront_cost.rsmeans_county")

# COMMAND ----------

# MAGIC %md ### Define weatherization costs by measure
# MAGIC Weatherization measures are the core components making up a weatherization package.

# COMMAND ----------

# Most estimated measure costs are drawn from Table 10 of the LBNL report (document p47).
# These costs reflect the cost of applying the measure to an archetypal home:  1,768 sq ft, 1-story, wood framed, single-family
# dwelling with a basement foundation, built in 1970.

ARCHETYPAL_HOME_SQFT = 1768  # sqftage used in study to translate between total costs and per sqftage costs

measure_schema = StructType(
    [
        StructField("measure", StringType(), False),
        StructField("cost_median_home", DoubleType(), True),
        StructField("cost_per_sqft", DoubleType(), True),
        # Attic type for applying attic floor insulation measures
        StructField("applicable_attic_types", ArrayType(StringType()), True),
        # Foundation type for applying foundation insulation measure
        StructField("applicable_foundation_types", ArrayType(StringType()), True),
        # Min and max ACH50 for applying light-touch air sealing measures (not applicable to other measures)
        StructField("ACH50_min", DoubleType(), True),
        StructField("ACH50_max", DoubleType(), True),
        # The list of weatherization packages that measure is used for
        StructField("weatherization_packages", ArrayType(StringType()), False),
    ]
)

# TODO: Define measures in dmlutils so we don't use raw strings in the `measure` field
costs_by_measure = spark.createDataFrame(
    [
        (
            "Attic Floor Insulation",
            4402.0,
            4402.0 / ARCHETYPAL_HOME_SQFT,
            ["Vented Attic"],
            None,
            None,
            None,
            [Weatherization.BASIC.value, Weatherization.ENHANCED.value, Weatherization.LIGHT_TOUCH_ENVELOPE.value],
        ),
        # General attic floor insulation up to IECC-Residential 2021 levels (R-30 to R-60 depending on climate zone, we use the cost for R60). General attic floor insulation only applies to vented attics.
        (
            "Wall Insulation",
            3135.0,
            3135.0 / ARCHETYPAL_HOME_SQFT,
            None,
            None,
            None,
            None,
            [Weatherization.BASIC.value, Weatherization.ENHANCED.value],
        ),  # Drill-and-fill insulation (R-13)
        (
            "Air Sealing",
            831.0,
            831.0 / ARCHETYPAL_HOME_SQFT,
            None,
            None,
            None,
            None,
            [Weatherization.BASIC.value, Weatherization.ENHANCED.value],
        ),  # General air sealing: reduction in ACH50 from ACH50 13 to ACH50 8.2 (~37% reduction)
        (
            "Duct Sealing",
            849.0,
            849.0 / ARCHETYPAL_HOME_SQFT,
            None,
            None,
            None,
            None,
            [Weatherization.BASIC.value, Weatherization.ENHANCED.value],
        ),
        # Duct sealing to 10% leakage, R-8 insulation (CFM25 97 post, which is about the same as 10% leakage
        # according to this https://basc.pnnl.gov/resource-guides/total-duct-leakage-tests#edit-group-description)
        (
            "Foundation Insulation",
            6794.0,
            6794.0 / ARCHETYPAL_HOME_SQFT,
            None,
            ["Vented Crawlspace", "Unvented Crawlspace", "Heated Basement"],
            None,
            None,
            [Weatherization.ENHANCED.value],
        ),
        # Add R-10 interior insulation to foundation walls and rim joists in conditioned basements and crawlspaces;
        # seal crawlspace vents (cost is for R18 insulation and not for sealing crawlspace vents)
        (
            "Roof Insulation",
            13575.0,
            13575.0 / ARCHETYPAL_HOME_SQFT,
            ["Finished Attic or Cathedral Ceilings"],
            None,
            None,
            None,
            [Weatherization.ENHANCED.value],
        ),  # For the Enhanced weatherization package only, insulate finished attics and cathedral ceilings to R-30
        (
            "Light Touch Attic Floor Insulation",
            4402.0,
            4402.0 / ARCHETYPAL_HOME_SQFT,
            ["Vented Attic", "Unvented Attic"],
            None,
            None,
            None,
            [Weatherization.LIGHT_TOUCH_AIR_SEALING_AND_ATTIC_INSULATION.value],
        ),
        # Light touch attic floor insulation up to IECC-Residential 2021 levels (R-49 or R-60 depending on climate zone, we use the cost for R60). Light touch  attic floor insulation applies to both vented and unvented attics.
        #
        # Some specific light-touch weatherization measures which we want are not included in Table 10.
        # For these we reference Appendix Figure G 96, which contains summary stats on the DER database (as opposed to using the archetypal home approach above).
        # Note that for the light touch weatherization package, which measure will be applied depends on the home's existing
        # ACH50 value. As a result, we don't aggregate up the measures; instead, specific measures will be joined to specific buildings below.
        (
            "Light Touch Air Sealing 40%",
            None,
            0.54,
            None,
            None,
            15.0,
            100.0,  # Placeholder value for highest possible ACH
            [
                Weatherization.LIGHT_TOUCH_AIR_SEALING.value,
                Weatherization.LIGHT_TOUCH_AIR_SEALING_AND_ATTIC_INSULATION.value,
            ],
        ),  # Light touch air sealing for dwelling units with greater than 15 ACH50: 40% total reduction in ACH50
        (
            "Light Touch Air Sealing 30%",
            None,
            0.40,
            None,
            None,
            10.0,
            15.0,
            [
                Weatherization.LIGHT_TOUCH_AIR_SEALING.value,
                Weatherization.LIGHT_TOUCH_AIR_SEALING_AND_ATTIC_INSULATION.value,
            ],
        ),  # Light touch air sealing for dwelling units with 10 < ACH50 <= 15: 30% total reduction in ACH50
        (
            "Light Touch Air Sealing 20%",
            None,
            0.34,
            None,
            None,
            7.5,
            10.0,
            [
                Weatherization.LIGHT_TOUCH_AIR_SEALING.value,
                Weatherization.LIGHT_TOUCH_AIR_SEALING_AND_ATTIC_INSULATION.value,
            ],
        ),  # Light touch air sealing for dwelling units with 7.5 < ACH50 <= 10: 20% total reduction in ACH50
    ],
    schema=measure_schema,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adjust for inflation
# MAGIC
# MAGIC The LBNL study was adjusted to 2019 values using RS Means, so we need to adjust from 2019 to
# MAGIC present-day costs. Note that RSMeans can be used to adjust for inflation as well as location.

# COMMAND ----------

# get the average construction price index in 2019
construction_price_index_2019 = (construction_inflation.where(F.col("year") == 2019).groupby().mean().collect())[0][
    "avg(construction_price_index)"
]

# get most recent month's price index
construction_price_index_present = construction_inflation.orderBy(F.desc("year"), F.desc("month")).collect()[0][
    "construction_price_index"
]

inflation_adjustment_factor = construction_price_index_present / construction_price_index_2019

# COMMAND ----------

# multiply by inflation adjustment factor
costs_by_measure_inflation_adjusted = costs_by_measure.withColumn(
    "cost_per_sqft", F.col("cost_per_sqft") * F.lit(inflation_adjustment_factor)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adjust for location
# MAGIC
# MAGIC The LBNL study was adjusted using RSMeans to develop national level costs, we now create
# MAGIC localized weatherization per square footage costs using RSMeans scaling factors

# COMMAND ----------

# First cross-join with counties to get an entry for every county x sq ft bin combo
costs_by_measure_inflation_adjusted = costs_by_measure_inflation_adjusted.crossJoin(resstock_counties)

# Join with RSMeans
costs_by_measure_inflation_location_adjusted = (
    costs_by_measure_inflation_adjusted.join(
        rsmeans.where((F.col("category") == "thermal & moisture protection") & (F.col("component") == "total")).select(
            "county_geoid", "rsmeans_price_index"
        ),
        on="county_geoid",
    )
    .withColumn("cost_per_sqft", F.col("cost_per_sqft") * F.col("rsmeans_price_index") / 100)
    .drop("rsmeans_price_index")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join to BuildingSet to get building-level cost predictions

# COMMAND ----------

# Join on county so each building gets a localized cost_per_sqft value
costs_by_building_measure = costs_by_measure_inflation_location_adjusted.join(
    building_set_metadata.select(
        "building_id",
        "county_geoid",
        "in_sqft",
        "in_geometry_attic_type",
        "in_geometry_foundation_type",
        "in_infiltration_ach50",
    ),
    on="county_geoid",
)

costs_by_building_measure = (
    # Apply light-touch air sealing measure logic
    costs_by_building_measure.filter(
        # Measures where applicable_attic_types, applicable_foundation_types, and ACH50 columns are Null should apply to all buildings
        (
            F.col("applicable_attic_types").isNull()
            & F.col("applicable_foundation_types").isNull()
            & F.col("ACH50_min").isNull()
            & F.col("ACH50_max").isNull()
        )
        # For Light touch air sealing, keep only rows where building's ACH50 matches measure ACH50 range
        | (
            (F.col("in_infiltration_ach50") > F.col("ACH50_min"))
            & (F.col("in_infiltration_ach50") <= F.col("ACH50_max"))
        )
        |
        # For attic floor and roof insulation, keep only rows where building's attic type is one of the measure attic types
        (
            F.array_contains(F.col("applicable_attic_types"), F.col("in_geometry_attic_type"))
            & (F.col("applicable_attic_types").isNotNull())
        )
        |
        # For foundation insulation, keep only rows where the building's foundation type is one of the measure foundation types
        (
            F.array_contains(F.col("applicable_foundation_types"), F.col("in_geometry_foundation_type"))
            & (F.col("applicable_foundation_types").isNotNull())
        )
    )
    # Calculate cost by building by multiplying by sqft
    .withColumn("upfront_cost_median", F.col("cost_per_sqft") * F.col("in_sqft")).drop(
        F.col("cost_per_sqft"), F.col("in_sqft")
    )
    # Cleanup
    .select("building_id", "measure", "upfront_cost_median")
)

# COMMAND ----------

# MAGIC %md ### Aggregate to building x package level
# MAGIC Now that we have inflation and location-adjusted measure costs by building, we need to aggregate the measures
# MAGIC for each building to get package costs by building. We retain measure costs as additional columns for downstream disaggregation as needed.

# COMMAND ----------

# Explode the dataframe of measures and packages so each row is a measure‑package combo
measures_to_packages = (
    costs_by_measure.select("measure", "weatherization_packages")
    .withColumn("weatherization_package", F.explode("weatherization_packages"))
    .drop("weatherization_packages")
)

# Join measures to building‑level costs
joined = costs_by_building_measure.join(measures_to_packages, on="measure", how="inner")

# Calculate total package cost by building
costs_by_building_package = joined.groupBy("building_id", "weatherization_package").agg(
    F.sum("upfront_cost_median").alias("upfront_cost_median")
)

# Pivot measure costs into separate columns
costs_by_building_measure_pivoted = (
    joined.groupBy("building_id", "weatherization_package").pivot("measure").agg(F.sum("upfront_cost_median"))
)
# Clean up column names
for col in costs_by_building_measure_pivoted.columns:
    if col not in ["building_id", "weatherization_package"]:
        costs_by_building_measure_pivoted = costs_by_building_measure_pivoted.withColumnRenamed(
            col, "upfront_cost_median_" + col.replace(" ", "_").lower()
        )

# Combine total and per‑measure columns
costs_by_building_package = costs_by_building_package.join(
    costs_by_building_measure_pivoted,
    on=["building_id", "weatherization_package"],
    how="inner",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write out table

# COMMAND ----------

# Specify BuildingSet in a table column and use selective overwrite for national BuildingSets.
costs_by_building_package = costs_by_building_package.withColumn("building_set", F.lit(BUILDING_SET))
write_options = {}
if BUILDING_SET in national_building_metadata:
    write_options = {"overwriteSchema": "false", "replaceWhere": f"building_set = '{BUILDING_SET}'"}

# Determine output table name based on building set
db_table_name = get_table_name_for_building_set(
    building_set=BUILDING_SET,
    catalog="processed",
    schema="upfront_costs",
    table_name="weatherization_cost_by_building_package",
)

# Write out table
sparkdb.write_delta_table(
    df=costs_by_building_package,
    db_table_name=db_table_name,
    mode=MODE,
    release_version=RELEASE_VERSION,
    **write_options,
)
