import pandas as pd
import numpy as np
import math
import statistics
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kurtosis
import more_itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.stats import skew
import os

def create_metafeaturedf(windowdf, features, windowsize):
    print('calculating df meta-features')
    windowdf['MetaFeatureDict'] = windowdf.apply(lambda row: calcmetafeatures(row.name, row.window, features, windowsize), axis=1)
    # windowdf['MetaFeatureVector'] = windowdf.apply(lambda row: (list(row.MetaFeatureDict.values())), axis=1)
    # windowdf['MetaFeatureVector'] = windowdf.apply(lambda row: [0 if math.isnan(x) else x for x in (row.MetaFeatureVector)], axis=1)
    # #windowdf['MetaFeatureVector'] = [MetaFeatureDict[x] for x in argv[4]]
    # metadf = windowdf.fillna(method="ffill")

    return windowdf

def expand_metafeaturedf(rawmetadf):
    df = pd.DataFrame(list(rawmetadf['MetaFeatureDict'])).set_index(rawmetadf.index)

    return df

def normalise(trainmetadf, contextcol):
    df = trainmetadf
    df.fillna(method="ffill")
    df.fillna(0)
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    # scaled[argv[3]] = list(trainmetadf[argv[3]])
    # scaled[contextcol] = list(trainmetadf[contextcol])

    return scaled

def calcmetafeatures(timestamp, window, features, windowsize):
    metafeatures = dict()
    filteredwindow = list(filter(None, window)) #TODO I Idont want to  have to filter
    # metafeatures['context'] = context #this puts the context as a meta-feature

    if 'mean' in features:
        metafeatures['mean'] = np.mean(filteredwindow)
    if 'std' in features:
        metafeatures['std'] = np.std(filteredwindow)
    if 'kurtosis' in features:
        metafeatures['kurtosis'] = kurtosis(filteredwindow)
    if 'adf' in features:
        metafeatures['adf'] = adfuller(filteredwindow)[0]
    if 'skew' in features:
        metafeatures['skew'] = skew(filteredwindow)
    if 'ac1' in features:
        acf = sm.tsa.acf(filteredwindow, nlags=3)
        metafeatures['ac1'] = acf[1]
    if 'ac2' in features:
        #acf = sm.tsa.acf(filteredwindow, nlags=3)
        metafeatures['ac2'] = acf[2]
    if 'max' in features:
        metafeatures['max'] = max(filteredwindow)
    if 'min' in features:
        metafeatures['min'] = min(filteredwindow)

    if 'decomposed' in features:
        stamprange = pd.date_range(timestamp - timedelta(hours=0, minutes=(windowsize - 1)), timestamp, freq="1T")
        data_tuples = list(zip(stamprange, filteredwindow))
        wts = pd.DataFrame(data_tuples, columns=['Time', 'point']).set_index('Time')
        metafeatures['decomposed'] = seasonal_decompose(wts, model='additive', extrapolate_trend='freq', period = 1)
    # Additive Decomposition #TODO gives me statsmodels objects
    # data_tuples = list(zip(usedstamps, filteredwindow))
    # wts = pd.DataFrame(data_tuples, columns=['Time', 'point']).set_index('Time')
    # metafeatures['decomposed'] = seasonal_decompose(wts, model='additive', extrapolate_trend='freq')

    return metafeatures