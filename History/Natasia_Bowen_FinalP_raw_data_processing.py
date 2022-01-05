import pandas as pd
import geopandas as gpd
import os
import censusdata
import numpy as np
import us
from statsmodels.formula.api import ols
import warnings

warnings.filterwarnings('ignore')

# set path 
#path = r'C:\Users\engel\Documents\GitHub\Data-II-Project'
#path = r'/Users/bowenli/Documents/GitHub/Data-II-Project'
path = r'C:/Users/ShrekTheOger/Documents/GitHub/Data-II-Project'

# set float display
pd.options.display.float_format = '{:.1f}'.format
us_contiguous = [state.name for state in us.STATES_CONTIGUOUS]

# references 
# https://towardsdatascience.com/mapping-us-census-data-with-python-607df3de4b9c
# https://pypi.org/project/CensusData/
# https://towardsdatascience.com/using-the-us-census-api-for-data-analysis-a-beginners-guide-98063791785c
# https://www.geeksforgeeks.org/how-to-create-dummy-variables-in-python-with-pandas/
# https://stackoverflow.com/questions/54757552/how-to-add-dummies-to-pandas-dataframe
# https://stackoverflow.com/questions/50733014/linear-regression-with-dummy-categorical-variables
# https://stackoverflow.com/questions/50733014/linear-regression-with-dummy-categorical-variables


# function for retrieving population data
def get_acs5_county_population(year):
    data = censusdata.download('acs5', year, 
                               censusdata.censusgeo([('county', '*')]),
                               ['B05002_001E', 'B05002_002E', 'B05002_003E', 
                                'B05002_004E', 'B05002_009E', 'B05002_013E'])
    
    # create dictionary for the column names
    column_names = {'B05002_001E': 'total_population'.capitalize(), 
                    'B05002_002E': 'total_native'.capitalize(), 
                    'B05002_003E': 'total_born_in_state'.capitalize(), 
                    'B05002_004E': 'total_born_out_state'.capitalize(),
                    'B05002_009E': 'total_born_outside_US'.capitalize(), 
                    'B05002_013E': 'total_foreign_born'.capitalize()}
    
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
    data['County'] = county_names
    data['State'] = state_names
    data['Year'] =  year
    data['County_id'] = county_ids
    data['State_id'] =  state_ids
    
    # rearrange column and replace column names 
    data = data[['Year', 'State', 'County', 'County_id', 'State_id',
                 'COUNTYFIPS', 'B05002_001E', 'B05002_002E', 'B05002_003E', 
                 'B05002_004E', 'B05002_009E', 'B05002_013E']]
    data = data.rename(column_names, axis=1) 
    
    # define data type
    data['State'] = data['State'].str.strip()
    
    return data

# function for retrieving income data
def get_acs5_county_income(year):
    data = censusdata.download('acs5', year, 
                               censusdata.censusgeo([('county', '*')]), 
                               ['B19301_001E'])
    
    # create dictionary for the column names
    column_names = {'B19301_001E': 'income_past12m'.capitalize()}
    
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


    # data.index = new_indices
    data['COUNTYFIPS'] = new_indices
    data['County'] = county_names
    data['State'] = state_names
    data['Year'] =  year
    data['County_id'] = county_ids
    data['State_id'] =  state_ids
    
    # rearrange column and replace column names 
    data = data[['Year', 'State', 'County', 'County_id', 'State_id',
                 'COUNTYFIPS', 'B19301_001E']]
    data = data.rename(column_names, axis=1) 
    
    # define data type
    data['State'] = data['State'].str.strip()
    
    return data

def get_population_df(ystart, yend):
    years = list(range(ystart, yend))
    df_population = pd.DataFrame()
    
    #retriving census data from year start to year end and merge into one dataframe
    for year in years: 
        population_year = get_acs5_county_population(year) 
        df_population = df_population.append(population_year)
    
    df_population.reset_index(drop = True, inplace = True)
    df_population = df_population[df_population['State'].isin(us_contiguous)]
    
    return df_population


def get_income_df(ystart, yend):
    years = list(range(ystart, yend))
    df_income = pd.DataFrame()

    #retriving census data from year start to year end and merge into one dataframe
    for year in years: 
        income_year = get_acs5_county_income(year) 
        df_income = df_income.append(income_year)
    
    df_income.reset_index(drop = True, inplace = True)
    df_income = df_income[df_income['State'].isin(us_contiguous)]
    
    return df_income

# Read Universities R&D Data based on ID 
# # https://stackoverflow.com/questions/45416684/python-pandas-replace-multiple-columns-zero-to-nan
def get_uni_fund(ystart, yend, nuni):
    fname = os.path.join(path+'/raw_data', 'NCSES', 'HERD_data_IPEDS.csv')
    data = pd.read_csv(fname, skiprows=10)
    
    years = list(range(ystart, yend))
    
    data = data.iloc[2:]
    col_names = {'Unnamed: 0': 'State', '<Fiscal Year>': 'IPEDSID'}
    data = data.rename(col_names, axis=1) 
    data = (data[data['IPEDSID'].str.contains('Total for selected values') 
                 == False])
    data = (data[data['IPEDSID'].str.contains
                 ('No match or exact match for IPEDS UnitID') == False])
    data.iloc[:,2:] = (data.iloc[:,2:].replace({'-':np.nan})
                       .replace(r',','',regex=True))
    
    data.iloc[:,2:] = data.iloc[:,2:].astype(float)
    data  = data.sort_values(['2010'], ascending = False)
    data = data.reset_index(drop = True)
    
    #Chooseing top n university as our observations
    data = data.head(nuni)

    data_filter = data[['State', 'IPEDSID', *[str(year) for year in years]]]
    
    rd_id_df = data_filter.melt(id_vars=["State", "IPEDSID"], 
                                var_name="Year",value_name="Fund")
    rd_id_df = rd_id_df[['Year','State', 'IPEDSID', 'Fund']]
    
    return rd_id_df

def get_regression_data(top_uni, df_variables): 
    
    # read university shape file 
    uni_us_shp = os.path.join(path+'/raw_data', 
                                'Colleges_and_Universities-shp', 
                                'Colleges_and_Universities.shp')
    uni_us = gpd.read_file(uni_us_shp)
    
    uni_filter = uni_us .loc[uni_us ['IPEDSID'].isin(top_uni['IPEDSID'])].copy()
    uni_filter = uni_filter[['IPEDSID', 'NAME', 'COUNTYFIPS']]

    # university fund 
    # matched university fund data with the population and income data in that county
    reg_data = top_uni.merge(uni_filter , how = 'inner', on = ['IPEDSID'])
    
    reg_data['Year'] = reg_data['Year'].astype(float)
    
    reg_data = reg_data.merge(df_variables, how = 'inner', 
                              on=['Year', 'State', 'COUNTYFIPS'])
    
    
    reg_data = reg_data[['Year', 'IPEDSID', 'Fund', 'Income_past12m', 
                    'Total_population', 'Total_born_out_state',
                    'Total_foreign_born']].copy()
    
    reg_data['Fund'] = reg_data['Fund'].astype(float) 
    
    return reg_data

def regression_result(y, x, df):
    regress = ols((f'{y} ~ {x} + C(Year2011) + C(Year2012) +'
                   'C(Year2013) + C(Year2014) + C(Year2015) + C(Year2016) +'
                   'C(Year2017) + C(Year2018) + C(Year2019)'), 
                  data=df).fit() 
    
    return regress


#merge all income and pop data on county level for the contiguous states in the US
pop_df = get_population_df(2010, 2020)
income_df = get_income_df(2010, 2020)
df = pop_df.merge(income_df, how = 'inner', 
                  on = ['Year','State',
                        'County', 'COUNTYFIPS', 'State_id', 'County_id'])

uni_df = get_uni_fund(2010, 2020, 50)

# export csv to local file
# the data frame would be use for graph
df.to_csv(path+'/refined_data/df_income_pop.csv', index=False, 
          encoding='utf-8') 
uni_df.to_csv(path+'/refined_data/uni_fund_df.csv', index = False, 
              encoding='utf-8')

reg_data = get_regression_data(uni_df, df)

reg_data_df = pd.get_dummies(reg_data,columns=['Year'], drop_first=True)

column_dummy = {'Year_2011.0': 'year2011'.capitalize(), 
                'Year_2012.0': 'year2012'.capitalize(), 
                'Year_2013.0': 'year2013'.capitalize(), 
                'Year_2014.0': 'year2014'.capitalize(),
                'Year_2015.0': 'year2015'.capitalize(), 
                'Year_2016.0': 'year2016'.capitalize(),
                'Year_2017.0': 'year2017'.capitalize(),
                'Year_2018.0': 'year2018'.capitalize(),
                'Year_2019.0': 'year2019'.capitalize()}

reg_data_df = reg_data_df.rename(column_dummy, axis=1)

fit_income = regression_result('Income_past12m', 'Fund', reg_data_df)
fit_income.summary()

fit_population = regression_result('Total_population', 'Fund', reg_data_df)
fit_population.summary()

fit_outstate = regression_result('Total_born_out_state ', 'Fund', reg_data_df)
fit_outstate.summary()

fit_foreigner = regression_result('Total_foreign_born', 'Fund', reg_data_df)
fit_foreigner.summary()

