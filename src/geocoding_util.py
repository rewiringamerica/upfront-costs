"""
Geocoding Helper Module.

This module provides a class, `CountyGeocoder`, for retrieving county FIPS codes.

The geocoding uses either Google's Geocoding API or the U.S. Census Bureau's Geocoding API.
It returns a dataframe of unique addresses or unique combinations of city and state with their
associated county FIPS codes.

To use this module:
1. Instantiate the `CountyGeocoder` class.
2. Use the `process_pyspark_dataframe` function to generate the dataframe of unique addresses and county fips codes.

Example:
    import pandas as pd
    from geocoding_util

    # Example dataframe
    data = [('New York, NY',), ('Los Angeles, CA',), ('Chicago, IL',)]
    columns = ["city_state"]
    df = spark.createDataFrame(data, columns)

    # Initialize the CountyGeocoder
    geocoder = geocoding_util.CountyGeocoder()

    # Generate the dataframe
    city_state_county = geocoding_util.process_pyspark_dataframe(df, 'city_state')


Classes:
    CountyGeocoder: Class to perform geocoding and retrieve county FIPS codes.
"""

import time

import pandas as pd
import requests


class CountyGeocoder:
    """
    A class to perform geocoding and retrieve county FIPS codes.

    Uses either Google's Geocoding API or the U.S. Census Bureau's Geocoding API.

    Attributes
    ----------
        api_key (str): The Google API key retrieved from Databricks secrets.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Google API key not found. Please pull the key from Databricks Secrets")

    def get_county_fips(self, address):
        """
        Retrieve the county FIPS code for a given city/state combination.

        Uses either Google's Geocoding API or the U.S. Census Bureau's geocoding API to find the county
        FIPS code for the given city/state combination in an address.

        Inputs:
        ------
            - address (str): A string containing an address, e.g., "1600 Amphitheatre Pkwy Mountain View, CA 94043".
            NOTE: address can also be a city/state combination, e.g., "New York, NY".

        Returns
        -------
        - str: Returns the county FIPS code if successful. If the geocoding fails, is incomplete,
                or if the county data is not found, returns None.
        """
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data["status"] == "OK":
                location = data["results"][0]["geometry"]["location"]
                lat = location["lat"]
                lon = location["lng"]
                try:
                    # Use the coordinates to get county FIPS code
                    census_url = f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates?x={lon}&y={lat}&benchmark=Public_AR_Census2020&vintage=Census2020_Census2020&layers=82&format=json"
                    census_response = requests.get(census_url)
                    census_response.raise_for_status()
                    census_data = census_response.json()

                    counties = census_data["result"]["geographies"].get("Counties")
                    if counties:
                        fips_code = counties[0]["GEOID"]
                        return fips_code
                    else:
                        return None
                except Exception as e:
                    print(f"Error occurred during Census API call: {e}")
                    return None

            else:
                return None
        except Exception as e:
            print(f"Error occurred during Google Geocoding API call: {e}")
            return None


def process_pyspark_dataframe(df, address_column, geocoder, sleep_seconds=1):
    """
    Process the given PySpark dataframe to generate a dataframe with county FIPS codes.

    Geocoding will be based on addresses or city/state combinations.

    Inputs:
    ------
        - df (pyspark.sql.DataFrame): The input PySpark dataframe containing a column with addresses
            or city/state combinations.
        - address_column (str): The name of the column in the dataframe containing the address or
            city/state combinations.
        - geocoder (class): Initialized CountyGeocoder.
        - sleep_seconds (int): the number of seconds to sleep between requests.

    Returns
    -------
    - pd.DataFrame: A pandas dataframe
    """
    # Collect unique addresses or city-state combos in a list
    address_list = [row[address_column] for row in df.select(address_column).distinct().collect()]

    # Empty list of rows
    rows = []

    # Loop through the list of city/state combos
    for address in address_list:
        fips = geocoder.get_county_fips(address)
        if fips is not None:
            # Append the new row as a dictionary to the list
            rows.append({"city_state": address, "county_geoid": fips})
            # Add in a step to sleep the code because previous requests were timing out
            time.sleep(sleep_seconds)

    # Create a DataFrame from the list of rows
    city_county_fips = pd.DataFrame(rows)

    return city_county_fips
