import pandas as pd
import geopandas as gpd
import os
import censusdata
import numpy as np
import us
from statsmodels.formula.api import ols
#import warnings
#warnings.filterwarnings('ignore')

# set path 
path = r'/Users/bowenli/Documents/GitHub/final-project-final-project-bowen-and-natasia'


# set float display
pd.options.display.float_format = '{:.2f}'.format
us_contiguous = [state.name for state in us.STATES_CONTIGUOUS]

# references 
# https://towardsdatascience.com/mapping-us-census-data-with-python-607df3de4b9c
# https://pypi.org/project/CensusData/
# https://towardsdatascience.com/using-the-us-census-api-for-data-analysis-a-beginners-guide-98063791785c
# https://www.geeksforgeeks.org/how-to-create-dummy-variables-in-python-with-pandas/
# https://stackoverflow.com/questions/54757552/how-to-add-dummies-to-pandas-dataframe
# https://stackoverflow.com/questions/50733014/linear-regression-with-dummy-categorical-variables
# https://stackoverflow.com/questions/50733014/linear-regression-with-dummy-categorical-variables
# https://www.python.org/dev/peps/pep-0008/#function-and-variable-names
# https://stackoverflow.com/questions/45416684/python-pandas-replace-multiple-columns-zero-to-nan

# function for retrieving population data
def get_acs5_county_population(year):
    data = censusdata.download('acs5', year, 
                               censusdata.censusgeo([('county', '*')]),
                               ['B05002_001E', 'B05002_002E', 'B05002_003E', 
                                'B05002_004E', 'B05002_009E', 'B05002_013E'])
    
    # create dictionary for the column names
    column_names = {'B05002_001E': 'total_population', 
                    'B05002_002E': 'total_native', 
                    'B05002_003E': 'total_born_in_state', 
                    'B05002_004E': 'total_born_out_state',
                    'B05002_009E': 'total_born_outside_US', 
                    'B05002_013E': 'total_foreign_born'}
    
    # create new column for county name and state name, 
    # the index would be FIPS codes 
    new_indices = []
    county_names = []
    state_names = []
    county_ids = []
    state_ids = []
    
    for index in data.index.tolist():
        new_index = index.geo[0][1] + index.geo[1][1]
        new_indices.append(new_index)
        county_name = index.name.split(',')[0]
        county_names.append(county_name)
        state_name = index.name.split(',')[1]
        state_names.append(state_name)
        state_id = index.geo[0][1]
        state_ids.append(state_id)
        county_id = index.geo[1][1]
        county_ids.append(county_id)

    data['COUNTYFIPS'] = new_indices
    data['county'] = county_names
    data['state'] = state_names
    data['year'] =  year
    data['county_id'] = county_ids
    data['state_id'] =  state_ids
    
    # rearrange column and replace column names 
    data = data[['year', 'state', 'county', 'county_id', 'state_id',
                 'COUNTYFIPS', 'B05002_001E', 'B05002_002E', 
                 'B05002_003E', 'B05002_004E', 'B05002_009E', 'B05002_013E']]
    data = data.rename(column_names, axis=1) 
    
    # define data type
    data['state'] = data['state'].str.strip()
    
    return data

# function for retrieving income data
def get_acs5_county_income(year):
    data = censusdata.download('acs5', year, 
                               censusdata.censusgeo([('county', '*')]), 
                               ['B19301_001E'])
    
    # create dictionary for the column names
    column_names = {'B19301_001E': 'income_past12m'}
    
    # create new column for county name and state name, the index would be FIPS codes 
    new_indices = []
    county_names = []
    state_names = []
    county_ids = []
    state_ids = []
    
    for index in data.index.tolist():
        new_index = index.geo[0][1] + index.geo[1][1]
        new_indices.append(new_index)
        county_name = index.name.split(',')[0]
        county_names.append(county_name)
        state_name = index.name.split(',')[1]
        state_names.append(state_name)
        state_id = index.geo[0][1]
        state_ids.append(state_id)
        county_id = index.geo[1][1]
        county_ids.append(county_id)

    data['COUNTYFIPS'] = new_indices
    data['county'] = county_names
    data['state'] = state_names
    data['year'] =  year
    data['county_id'] = county_ids
    data['state_id'] =  state_ids
    
    # rearrange column and replace column names 
    data = data[['year', 'state', 'county', 'county_id', 'state_id',
                 'COUNTYFIPS', 'B19301_001E']]
    data = data.rename(column_names, axis=1) 
    
    # define data type
    data['state'] = data['state'].str.strip()
    
    return data

# function for retreiving muiltiple years of population data
def get_population_df(ystart, yend):
    years = list(range(ystart, yend))
    df_population = pd.DataFrame()
    
    #retriving census data from year start to year end and merge into one dataframe
    for year in years: 
        population_year = get_acs5_county_population(year) 
        df_population = df_population.append(population_year)
    
    df_population.reset_index(drop = True, inplace = True)
    df_population = df_population[df_population['state'].isin(us_contiguous)]
    
    return df_population


def get_income_df(ystart, yend):
    years = list(range(ystart, yend))
    df_income = pd.DataFrame()

    #retriving census data from year start to year end and merge into one dataframe
    for year in years: 
        income_year = get_acs5_county_income(year) 
        df_income = df_income.append(income_year)
    
    df_income.reset_index(drop = True, inplace = True)
    df_income = df_income[df_income['state'].isin(us_contiguous)]
    
    return df_income

# Read Universities R&D Data based on ID 
def get_uni_fund(ystart, yend, nuni):
    fname = os.path.join(path+'/raw_data', 'NCSES', 'HERD_data_IPEDS.csv')
    data = pd.read_csv(fname, skiprows=10)
    
    years = list(range(ystart, yend))
    
    data = data.iloc[2:]
    col_names = {'Unnamed: 0': 'state', '<Fiscal Year>': 'IPEDSID'}
    data = data.rename(col_names, axis=1) 
    
    # remove all the data that has no university ID 
    # and remove total data to avoid double counting
    data = (data[data['IPEDSID'].str.contains('Total for selected values') 
                 == False])
    data = (data[data['IPEDSID'].str.contains
                 ('No match or exact match for IPEDS UnitID') == False])
    data.iloc[:,2:] = (data.iloc[:,2:].replace({'-':np.nan})
                       .replace(r',','',regex=True))
    
    data.iloc[:,2:] = data.iloc[:,2:].astype(float)
    data  = data.sort_values(['2010'], ascending = False)
    data = data.reset_index(drop = True)
    
    # choose top n university as our observations
    data = data.head(nuni)

    data_filter = data[['state', 'IPEDSID', *[str(year) for year in years]]]
    
    rd_id_df = data_filter.melt(id_vars=['state', 'IPEDSID'], 
                                var_name='year', value_name='fund')
    rd_id_df = rd_id_df[['year','state', 'IPEDSID', 'fund']]
    
    return rd_id_df

def get_regression_data(top_uni, df_variables): 
    # to get regression data we first to read and match the county level data
    # with the university data that contains county information
    # read university shape file 
    uni_us_shp = os.path.join(path+'/raw_data', 
                              'Colleges_and_Universities-shp', 
                              'Colleges_and_Universities.shp')
    uni_us = gpd.read_file(uni_us_shp)
    
    # filter university based on the lis of top universities
    uni_filter = uni_us .loc[uni_us ['IPEDSID'].isin(top_uni['IPEDSID'])].copy()
    uni_filter = uni_filter[['IPEDSID', 'NAME', 'COUNTYFIPS']]

    # university fund 
    # matched university fund data with the population and income data in that county
    reg_data = top_uni.merge(uni_filter , how = 'inner', on = ['IPEDSID'])
    
    reg_data['year'] = reg_data['year'].astype('int')
    
    # merge with data frame of income and population
    reg_data = reg_data.merge(df_variables, how = 'inner', 
                              on=['year', 'state', 'COUNTYFIPS'])
    
    
    reg_data = reg_data[['year', 'IPEDSID', 'fund', 'income_past12m', 
                         'total_population', 'total_born_out_state',
                         'total_foreign_born']].copy()
    
    reg_data['fund'] = reg_data['fund'].astype(float) 
    
    return reg_data

# create function for regression
def regression_result(y, x, df):
    regress = ols((f'{y} ~ {x} + C(year_2011) + C(year_2012) +'
                   'C(year_2013) + C(year_2014) + C(year_2015) + '
                   'C(year_2016) + C(year_2017) + C(year_2018) + '
                   'C(year_2019)'), 
                  data=df).fit() 
    
    return regress


# merge income and pop data on county level for the contiguous states in the US
pop_df = get_population_df(2010, 2020)
income_df = get_income_df(2010, 2020)
df = pop_df.merge(income_df, how = 'inner', 
                  on = ['year','state',
                        'county', 'COUNTYFIPS', 'state_id', 'county_id'])

# get top 50 universities in USA
uni_df = get_uni_fund(2010, 2020, 50)

# export csv to local file
# the data frame would be use for graph
df.to_csv(path+'/refined_data/df_income_pop.csv', index=False, 
          encoding='utf-8') 
uni_df.to_csv(path+'/refined_data/uni_fund_df.csv', index = False, 
              encoding='utf-8')


## regression analysis
# get the data for regression 
reg_data = get_regression_data(uni_df, df)

## create summary statistics
# average of R&D fund over the last 10 years
summary_stats = reg_data.groupby('year').describe()

# create dummy variables for year
reg_data_dummy = pd.get_dummies(reg_data,columns=['year'], drop_first=True)

# regress income on fund
fit_income = regression_result('income_past12m', 'fund', reg_data_dummy)
fit_income.summary()

# regress total population on fund
fit_population = regression_result('total_population', 'fund', reg_data_dummy)
fit_population.summary()

# regress total population from out of state on fund
fit_outstate = regression_result('total_born_out_state ', 'fund', 
                                 reg_data_dummy)
fit_outstate.summary()

# regress total foreign population on fund
fit_foreigner = regression_result('total_foreign_born', 'fund', reg_data_dummy)
fit_foreigner.summary()

