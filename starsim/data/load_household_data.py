"""
Load country household size from UN data source.

Returns dictionary with country name as key and average household size as value,
for insertion in starsim/data.
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import requests
from io import BytesIO
from utils import get_country_aliases

# URL of the Excel file
url = 'https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/undesa_pd_2022_hh-size-composition.xlsx'

# Define headers with a user-agent
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# Send the GET request
response = requests.get(url, headers=headers)
df = None
# Check if the request was successful
if response.status_code == 200:
    # Read the content into a BytesIO object
    excel_data = BytesIO(response.content)
    # Load the Excel file into a Pandas DataFrame
    df = pd.read_excel(excel_data, sheet_name=None)  # sheet_name=None reads all sheets
    # Display the sheet names
    print(df.keys())
else:
    print(f"Failed to download file. Status code: {response.status_code}")



df_raw = pd.read_excel(excel_data, sheet_name='HH size and composition 2022', skiprows=4)

assert len(df_raw) > 1, "Loaded UN Household size."

# Select and rename columns
target_columns = ['Country or area', 'Reference date (dd/mm/yyyy)', 'Average household size (number of members)']
df = df_raw[target_columns].copy()
df.columns = ['country', 'date', 'size']

# Convert date column to datetime type and replace nodata with NA.
df['date'] = df['date'].apply(lambda d: pd.to_datetime(d, format='%d/%m/%Y'))
df['size'] = df['size'].apply(lambda s: np.nan if isinstance(s, str) and s == '..' else s)
df = df.dropna()

# Take the most recent household size for each country.
df = df.sort_values(by=['country', 'date']).groupby(by=['country']).last()[['size']]
un_country_households_dict = df.to_dict()['size'].copy()

# Add to the dictionary commonly used aliases like USA for United States of America.
us_countries = [k.lower() for k in un_country_households_dict]
country_mappings_dict = get_country_aliases()


for alias, name in country_mappings_dict.items():
    if name.lower() in us_countries:
        un_country_households_dict[alias] = un_country_households_dict[name]

pathname = str(sc.thisdir(ss.__file__, aspath=True) / 'data' / 'household_size_data.py')
print(f'To update, copy the data below into the following file:\n{pathname}\n')
print(un_country_households_dict)