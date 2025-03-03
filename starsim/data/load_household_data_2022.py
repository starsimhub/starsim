import pandas as pd
import requests
from io import BytesIO
from pprint import pprint

def download_excel(url = 'https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/undesa_pd_2022_hh-size-composition.xlsx'):
    # Download the Excel file from the given URL
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        excel_data = BytesIO(response.content)  
        return pd.ExcelFile(excel_data)
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

def load_and_clean_hh_data(xls):
    """
    Reads and cleans the 'Household Size and Composition 2022' dataset.
    Returns a cleaned Pandas DataFrame and a dictionary for easy indexing.
    """
    # Load relevant sheet and skip initial metadata rows
    df = pd.read_excel(xls, sheet_name="HH size and composition 2022", skiprows=4)

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Selecting relevant columns for household composition analysis
    selected_columns = {
        "Country or area": "Country",
        "ISO Code": "ISO Code",
        "Data source category": "Data Source Category",
        "Reference date (dd/mm/yyyy)": "Reference Date",
        "Average household size (number of members)": "Average Household Size",
        "1 member": "Households 1 Member",
        "2-3 members": "Households 2-3 Members",
        "4-5 members": "Households 4-5 Members",
        "6 or more members": "Households 6+ Members",
        "Female head of household (percentage of households)": "Female Head of Household",
        "Couple with children": "Couple with Children",
        "Couple only": "Couple without Children",
        "Single mother with children": "Single Mother with Children",
        "Single father with children": "Single Father with Children",
        "Extended family": "Extended Family",
        "Non-relatives": "Non-Relatives",
        "Unknown": "Unknown Household Type",
        "Nuclear": "Nuclear Household",
        "Multi-generation": "Multi-Generation Household",
        "Three generation": "Three Generation Household",
        "Skip generation": "Skip Generation Household"
    }

    # Apply column selection and renaming
    df = df[list(selected_columns.keys())].rename(columns=selected_columns)

    # Remove any rows where "Country" is NaN (to eliminate unnecessary artifacts)
    df = df.dropna(subset=["Country"])

    # Convert Reference Date to datetime format
    df["Reference Date"] = pd.to_datetime(df["Reference Date"], errors="coerce")

    # Convert numerical columns to proper data types
    num_columns = df.columns[4:]  # Household-related data
    df[num_columns] = df[num_columns].apply(pd.to_numeric, errors='coerce')

    # Sort by Reference Date to get the most recent entry per country
    df = df.sort_values(by=["Country", "Reference Date"], ascending=[True, False])

    # Keep only the latest record for each country
    df = df.drop_duplicates(subset=["Country"], keep="first")

    # Reset index after deduplication
    df = df.reset_index(drop=True)

    # Create a dictionary with unique country indexing for easy lookups
    hh_data_dict = df.set_index("Country").to_dict(orient="index")

    # Return the cleaned DataFrame and the Countries dictionary
    return df, hh_data_dict


# Example of 'manual' usage:
if __name__ == "__main__":
    try:
        xls = download_excel()
        cleaned_df, household_data_dict = load_and_clean_hh_data(xls)
        # Display the first few rows of the cleaned DataFrame
        print(cleaned_df.head())

        # Example: Access data for a specific country
        country_name = "Mexico"
        if country_name in household_data_dict:
            print(f"Household data for {country_name}:")
            pprint(household_data_dict[country_name])
        else:
            print(f"No data found for {country_name}")
            pprint(list(household_data_dict.keys()))
    except Exception as e:
        print(f"Error: {e}")
