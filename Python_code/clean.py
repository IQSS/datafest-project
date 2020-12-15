## DAY 1

from pyDataverse.api import Api
from pyDataverse.models import Dataverse
import pandas as pd
import numpy as np
import requests
from functools import reduce
import matplotlib.pyplot as plt
import math

## Acquring data from APIs
# establish connection
base_url = 'https://dataverse.harvard.edu/'
api = Api(base_url)
print(api.status)

# get the digital object identifier for the Harvard Dataverse dataset
DOI = "doi:10.7910/DVN/HIDLTK"

# retrieve the contents of the dataset
covid = api.get_dataset(DOI)

covid_files_list = covid.json()['data']['latestVersion']['files']

for fileObject in covid_files_list:
    print("File name is {}; id is {}".format(fileObject["dataFile"]["filename"], fileObject["dataFile"]["id"]))

# get data file
US_states_cases_file = api.get_datafile("4201597")

# convert
in_text = US_states_cases_file.content
tmp = "US_states_cases.tab"

f = open(tmp, "wb")
f.write(in_text)
f.close()

US_states_cases = pd.read_csv(tmp, sep='\t')

print(US_states_cases.head(10))

## Cleaning data
# select columns of interest
US_states_cases_filtered = US_states_cases.filter(regex="^\\d", axis=1)
US_states_cases_selected = US_states_cases.loc[:, ['fips', 'NAME', 'POP10']]
US_states_cases_selected = US_states_cases_selected.assign(**US_states_cases_filtered)
# rename some columns
US_states_cases_selected = US_states_cases_selected.rename(columns={'fips':'GEOID', 'NAME':'state', 'POP10':'pop_count_2010'})
# reshape to long format for dates
US_states_cases_selected = pd.melt(US_states_cases_selected, id_vars=["GEOID", "state", "pop_count_2010"], var_name='date', value_name="cases_cum")
US_states_cases_selected = US_states_cases_selected.sort_values(['GEOID', 'date']).reset_index(drop=True)
# create new derived time variables from dates 
US_states_cases_selected["day_of_year"] = pd.to_datetime(US_states_cases_selected.date).dt.dayofyear
US_states_cases_selected["week_of_year"] = pd.to_datetime(US_states_cases_selected.date).dt.isocalendar().week
US_states_cases_selected["month"] = pd.to_datetime(US_states_cases_selected.date).dt.month
# create cases counts
US_states_cases_selected["cases_count"] = US_states_cases_selected.groupby('state').cases_cum.apply(lambda x: x - x.shift(1)).fillna(0)
# tidy-up negative counts
US_states_cases_selected["cases_count_pos"] = np.where(US_states_cases_selected["cases_count"] < 0, 0, US_states_cases_selected["cases_count"])
# create cases rates
US_states_cases_selected["cases_rate_100K"] = (US_states_cases_selected["cases_count_pos"] / US_states_cases_selected["pop_count_2010"]) * 1e5
US_states_cases_selected["cases_cum_rate_100K"] = (US_states_cases_selected["cases_cum"] / US_states_cases_selected["pop_count_2010"]) * 1e5

#US_states_cases_selected.to_csv("US_states_cases_selected.csv")

# Aggregate data
# aggregate to weekly level (for later modeling)
aggs_by_col = {'pop_count_2010': lambda x: np.mean(x), 'cases_cum': 'sum', 'cases_cum_rate_100K': 'sum', 'cases_count_pos': 'sum', 'cases_rate_100K': 'sum'}

US_states_cases_week = US_states_cases_selected.groupby(['GEOID', 'state', 'week_of_year'], as_index=False).agg(aggs_by_col)
print(US_states_cases_week.head(60))

#US_states_cases_week.to_csv("US_states_cases_week.csv")

# OPTIONAL: U.S. Census data
# store API key
API_key = "77498e49cff2f881427082a66a00f284f0590ba7"

# access the Population Estimates API and extract variables of interest
# provides overall population estimates and population densities
pop_url = f'https://api.census.gov/data/2019/pep/population?get=NAME,POP,DENSITY&for=state:*&key={API_key}'
response = requests.get(pop_url)
pop_data = response.json()
pop_df = pd.DataFrame(pop_data[1:], columns=pop_data[0]).rename(columns={'NAME':'state', 'state':'GEOID'})
print(pop_df.head(60))

# get population estimates by age group
age_url = f'https://api.census.gov/data/2019/pep/charagegroups?get=NAME,AGEGROUP,POP&for=state:*&key={API_key}'
response = requests.get(age_url)
age_data = response.json()
age_df = pd.DataFrame(age_data[1:], columns=age_data[0]).rename(columns={'NAME':'state', 'state':'GEOID'})
print(age_df.head(60))

# get population estimates by sex
sex_url = f'https://api.census.gov/data/2019/pep/charagegroups?get=NAME,SEX,POP&for=state:*&key={API_key}'
response = requests.get(sex_url)
sex_data = response.json()
sex_df = pd.DataFrame(sex_data[1:], columns=sex_data[0]).rename(columns={'NAME':'state', 'state':'GEOID'})
print(sex_df.head(60))

# get population estimates by race
race_url = f'https://api.census.gov/data/2019/pep/charagegroups?get=NAME,RACE,POP&for=state:*&key={API_key}'
response = requests.get(race_url)
race_data = response.json()
race_df = pd.DataFrame(race_data[1:], columns=race_data[0]).rename(columns={'NAME':'state', 'state':'GEOID'})
print(race_df.head(60))

# clean overall population estimates
# order by GEOID (same as state FIPS code)
pop_df[["GEOID", "POP"]] = pop_df[["GEOID", "POP"]].apply(pd.to_numeric)
pop_wide = pop_df.sort_values(['GEOID']).reset_index(drop=True)
# exclude Puerto Rico, rename, and no need to reshape to wide format
pop_wide = pop_wide[pop_wide.state != "Puerto Rico"].rename(columns={'POP':'pop_count_2019', 'DENSITY':'pop_density_2019'})
print(pop_wide.head(60))

# order by GEOID (same as state FIPS code)
age_df[["GEOID", "AGEGROUP", "POP"]] = age_df[["GEOID", "AGEGROUP", "POP"]].apply(pd.to_numeric)
age_df = age_df.sort_values(['GEOID', 'AGEGROUP']).reset_index(drop=True)
# reshape the age groups to wide format
age_wide = age_df.pivot_table(index=["GEOID", "state"], columns='AGEGROUP', values="POP").reset_index()
# create variable for percentortion of people that are 65 years and older
age_wide["percent_age65over"] = (age_wide[26] / age_wide[0]) * 100
# select columns of interest
age_wide = age_wide.loc[:, ['GEOID', 'state', 'percent_age65over']]
# exclude Puerto Rico
age_wide = age_wide[age_wide.state != "Puerto Rico"]
print(age_wide.head(60))

# order by GEOID (same as state FIPS code)
sex_df[["GEOID", "SEX", "POP"]] = sex_df[["GEOID", "SEX", "POP"]].apply(pd.to_numeric)
sex_df = sex_df.sort_values(['GEOID', 'SEX']).reset_index(drop=True)
# reshape the sex groups to wide format
sex_wide = sex_df.pivot_table(index=["GEOID", "state"], columns='SEX', values="POP").reset_index()
# create variable for percentortion of people that are female
sex_wide["percent_female"] = (sex_wide[2] / sex_wide[0]) * 100
# select columns of interest
sex_wide = sex_wide.loc[:, ['GEOID', 'state', 'percent_female']]
# exclude Puerto Rico
sex_wide = sex_wide[sex_wide.state != "Puerto Rico"]
print(sex_wide.head(60))

# order by GEOID (same as state FIPS code)
race_df[["GEOID", "RACE", "POP"]] = race_df[["GEOID", "RACE", "POP"]].apply(pd.to_numeric)
race_df = race_df.sort_values(['GEOID', 'RACE']).reset_index(drop=True)
# reshape the race categories to wide format
race_wide = race_df.pivot_table(index=["GEOID", "state"], columns='RACE', values="POP").reset_index()
# create variables for percentages of people that are black and white
race_wide["percent_white"] = (race_wide[1] / race_wide[0]) * 100
race_wide["percent_black"] = (race_wide[2] / race_wide[0]) * 100
# select columns of interest
race_wide = race_wide.loc[:, ['GEOID', 'state', 'percent_white', 'percent_black']]
# exclude Puerto Rico
race_wide = race_wide[race_wide.state != "Puerto Rico"]
print(race_wide.head(60))

# merge all the cleaned Census data into one object called demographics
data_frames = [pop_wide, age_wide, sex_wide, race_wide]
demographics = reduce(lambda  left,right: pd.merge(left,right,on=['GEOID', 'state'], how='left'), data_frames)
print(demographics.head(60))

# merge the COVID-19 cases data with Census demographic data
data_frames = [US_states_cases_selected, demographics]
US_cases_long_demogr = reduce(lambda  left,right: pd.merge(left,right,on=['GEOID', 'state'], how='left'), data_frames)

# update the case rate variables to use population estimates from 2019
US_cases_long_demogr["cases_cum_rate_100K"] = (US_cases_long_demogr["cases_cum"] / US_cases_long_demogr["pop_count_2019"]) * 1e5
US_cases_long_demogr["cases_rate_100K"] = (US_cases_long_demogr["cases_count_pos"] / US_cases_long_demogr["pop_count_2019"]) * 1e5
print(US_cases_long_demogr.head(60))

# aggregate to the weekly-level
aggs_by_col = {'pop_count_2019': lambda x: np.mean(x), 'percent_age65over': lambda x: np.mean(x), 'percent_female': lambda x: np.mean(x), \
    'percent_white': lambda x: np.mean(x), 'percent_black': lambda x: np.mean(x),'cases_cum': 'sum', 'cases_cum_rate_100K': 'sum', \
    'cases_count_pos': 'sum', 'cases_rate_100K': 'sum'}

US_cases_long_demogr_week = US_cases_long_demogr.groupby(['state', 'week_of_year'], as_index=False).agg(aggs_by_col)
print(US_cases_long_demogr_week.head(60))

# store the data frame in a .csv file
US_cases_long_demogr_week.to_csv("US_cases_long_demogr_week.csv")

