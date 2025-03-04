import requests
import pandas as pd
import os
from pprint import pprint
# World Bank API base URL
WB_API_URL = "http://api.worldbank.org/v2"

def get_country_codes():
    """
    Fetches all valid country codes from the World Bank API.
    
    :return: Dictionary mapping country codes to country names.
    """
    url = f"{WB_API_URL}/country?format=json&per_page=300"
    response = requests.get(url)

    if response.status_code == 200:
        json_data = response.json()
        if json_data and isinstance(json_data, list) and len(json_data) > 1:
            countries = {item["id"]: item["name"] for item in json_data[1] if item["region"]["id"] != "NA"}
            return countries
        else:
            print("Invalid country code provided, please see the list above.")
    else:
        print(f"Error fetching country codes: {response.status_code}")
        pprint(response.json())
    # Return an empty dictionary if the request fails
    return {}

def fetch_world_bank_data(country_code):
    """
    Fetches age distribution, birth rates, and death rates from World Bank API for a given country.
    
    :param country_code: ISO 3166-1 alpha-3 country code (e.g., "USA" for the United States).
    :return: Pandas DataFrame with demographic data.
    """
    # List of key demographic indicators
    indicators = {
        "SP.POP.0004.TO.ZS": "Age 0-4 (% of total)",
        "SP.POP.0509.TO.ZS": "Age 5-9 (% of total)",
        "SP.POP.1014.TO.ZS": "Age 10-14 (% of total)",
        "SP.POP.1519.TO.ZS": "Age 15-19 (% of total)",
        "SP.POP.2024.TO.ZS": "Age 20-24 (% of total)",
        "SP.POP.2529.TO.ZS": "Age 25-29 (% of total)",
        "SP.POP.3034.TO.ZS": "Age 30-34 (% of total)",
        "SP.POP.3539.TO.ZS": "Age 35-39 (% of total)",
        "SP.POP.4044.TO.ZS": "Age 40-44 (% of total)",
        "SP.POP.4549.TO.ZS": "Age 45-49 (% of total)",
        "SP.POP.5054.TO.ZS": "Age 50-54 (% of total)",
        "SP.POP.5559.TO.ZS": "Age 55-59 (% of total)",
        "SP.POP.6064.TO.ZS": "Age 60-64 (% of total)",
        "SP.POP.6569.TO.ZS": "Age 65-69 (% of total)",
        "SP.POP.7074.TO.ZS": "Age 70-74 (% of total)",
        "SP.POP.7579.TO.ZS": "Age 75-79 (% of total)",
        "SP.POP.80UP.TO.ZS": "Age 80+ (% of total)",
        "SP.DYN.CBRT.IN": "Crude Birth Rate (per 1,000 people)",
        "SP.DYN.CDRT.IN": "Crude Death Rate (per 1,000 people)"
    }

    base_url = f"{WB_API_URL}/country/{{}}/indicator/{{}}?format=json&per_page=100"
    
    data = {}
    for code, name in indicators.items():
        url = base_url.format(country_code, code)
        response = requests.get(url)
        
        if response.status_code == 200:
            json_data = response.json()
            if json_data and isinstance(json_data, list) and len(json_data) > 1:
                values = json_data[1]
                if values:
                    latest_data = next((item for item in values if item["value"] is not None), None)
                    if latest_data:
                        data[name] = latest_data["value"]
                    else:
                        data[name] = "No Data Available"
                else:
                    data[name] = "No Data Available"
        else:
            data[name] = "API Error"

    # Convert to DataFrame
    df = pd.DataFrame(data, index=[country_code])
    return df

def save_to_file(df, country_code):
    """
    Saves the DataFrame to a CSV file with the full path of the running script.
    
    :param df: Pandas DataFrame to save.
    :param country_code: Country code used for filename.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
    filename = f"world_bank_data_{country_code}.csv"
    file_path = os.path.join(script_dir, filename)
    
    df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")

# Entry point
if __name__ == "__main__":
    # country = input("Enter a country code (ISO 3166-1 alpha-3, e.g., USA, IND, BRA): ").upper()
    country = "COD"
    # Get valid country codes
    valid_countries = get_country_codes()

    # Validate input
    if country not in valid_countries:
        print(f"Invalid country code: {country}. Here are valid options:")
        for code, name in valid_countries.items():
            print(f"{code}: {name}")
    else:
        # Fetch data
        df = fetch_world_bank_data(country)
        print(df)
        
        # Save to file
        save_to_file(df, country)
