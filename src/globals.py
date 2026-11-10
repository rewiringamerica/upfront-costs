import pandas as pd
from cloudpathlib import CloudPath

# -- Paths -- #

# gcloud
BUCKET_FPATH = CloudPath("gs://the-cube")
DATA_FPATH = BUCKET_FPATH / "data"
RAW_DATA_FPATH = DATA_FPATH / "raw"
BSB_DATA_FPATH = RAW_DATA_FPATH / "bsb_sims"
PROCESSED_DATA_FPATH = DATA_FPATH / "processed"
PREPARED_DATA_FPATH = DATA_FPATH / "prepared"
EXPORT_FPATH = BUCKET_FPATH / "export"

SHAPEFILE_FPATH = PROCESSED_DATA_FPATH / "geospatial" / "shapefile"

# -- Geography -- #
# Dataframe of State_name, State_abbr, State_GEOID
STATE_GEOIDS = pd.read_csv(str(RAW_DATA_FPATH / "geospatial" / "state_geoids.csv"), dtype={"state_geoid": str})
# GEOIDs for Territories
NON_STATE_GEOIDS = ["60", "66", "69", "72", "78"]
