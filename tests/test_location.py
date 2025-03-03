import pytest
import pandas as pd
from starsim.data.load_household_data_2022 import load_and_clean_hh_data, download_excel
import os

def test_load_and_clean_hh_data():
    xls = download_excel()
    cleaned_df, household_data_dict = load_and_clean_hh_data(xls)
    # Display the first few rows of the cleaned DataFrame
    print(cleaned_df.head())
    # Check if the DataFrame is cleaned correctly
    assert not cleaned_df.empty
    assert "Country" in cleaned_df.columns
    assert "Average Household Size" in cleaned_df.columns
    assert cleaned_df["Country"].is_unique
    # Check if the dictionary is created correctly
    assert "Albania" in household_data_dict
    assert household_data_dict["Albania"]["ISO Code"] == 8

def test_data_integrity():
    xls = download_excel()
    cleaned_df, household_data_dict = load_and_clean_hh_data(xls)
    # Check if the data types are correct
    assert cleaned_df["Country"].dtype == object
    assert cleaned_df["ISO Code"].dtype == int
    assert cleaned_df["Average Household Size"].dtype == float
    # Check if the Reference Date is in datetime format
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["Reference Date"])
