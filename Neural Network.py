from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

features = [
       #'IGCCCONTRIBUTION_UP', 'IGCCCONTRIBUTION_DOWN', 'UPWARD_DISPATCH_x',
       # 'DOWNWARD_DISPATCH_x', 'RESERVE_UPWARD_DISPATCH',
       # 'RESERVE_DOWNWARD_DISPATCH', 'INCIDENT_RESERVE_UP_INDICATOR',
       #'INCIDENT_RESERVE_DOWN_INDICATOR',
       'MID_PRICE',
       # 'Actual Load', 'BIOMASS_AGGREGATED', 'BIOMASS_CONSUMPTION',
       # 'FOSSIL_GAS_AGGREGATED', 'FOSSIL_GAS_CONSUMPTION',
       # 'FOSSIL_COAL_AGGREGATED', 'FOSSIL_COAL_CONSUMPTION', 'HYDRO_AGGREGATED',
       # 'HYDRO_CONSUMPTION', 'NUCLEAR_AGGREGATED', 'NUCLEAR_CONSUMPTION',
       # 'OTHER_AGGREGATED', 'OTHER_CONSUMPTION', 'SOLAR_AGGREGATED',
       # 'SOLAR_CONSUMPTION', 'WASTE_AGGREGATED', 'WASTE_CONSUMPTION',
       # 'WIND_OFFSHORE_AGGREGATED', 'WIND_OFFSHORE_CONSUMPTION',
       # 'WIND_ONSHORE_AGGREGATED', 'WIND_ONSHORE_CONSUMPTION',
       # 'CROSS_FLOW_DE_NL', 'CROSS_FLOW_NO_NL', 'CROSS_FLOW_BE_NL',
       # 'CROSS_FLOW_GB_NL', 'UPWARD_DISPATCH_y', 'DOWNWARD_DISPATCH_y',
       # 'INCENTIVE_COMPONENT', 'TAKE_FROM_SYSTEM', 'FEED_INTO_SYSTEM',
       'REGULATION_STATE',
       #'Forecasted Load', 'GENERATION_FORECAST',
       # 'SOLAR_FORECAST', 'WIND_OFFSHORE_FORECAST', 'WIND_ONSHORE_FORECAST',
       #'PRICE_FORECAST'
    ]
preclustertrain = pd.read_pickle('../Clustering/CleanData/Minutely/2018')
train_clustered = pd.read_pickle('ClusteredData/Train')(n=10000)
df = preclustertrain[preclustertrain.index.isin(train_clustered.index)]
df['cluster'] = list(train_clustered['cluster'])

def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans
num_cols = get_cols_with_no_nans(df , 'num')
cat_cols = get_cols_with_no_nans(df , 'no_num')