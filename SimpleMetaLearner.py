import pandas as pd
import seaborn as sns
import xgboost as xgb
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from SimpleUtils import *
import os

features = [
    'Target',
    # 'Target_STANDARDIZED',
    'REGULATION_STATE_-1',
    'REGULATION_STATE_0', 'REGULATION_STATE_1', 'REGULATION_STATE_2',
       # 'IGCCCONTRIBUTION_UP', 'IGCCCONTRIBUTION_DOWN', 'UPWARD_DISPATCH_x',
       # 'DOWNWARD_DISPATCH_x', 'MID_PRICE', 'MIN_PRICE', 'MAX_PRICE',
       # 'Actual Load', 'FOSSIL_GAS_AGGREGATED', 'FOSSIL_GAS_CONSUMPTION',
       # 'FOSSIL_COAL_AGGREGATED', 'FOSSIL_COAL_CONSUMPTION',
       # 'NUCLEAR_AGGREGATED', 'NUCLEAR_CONSUMPTION', 'OTHER_AGGREGATED',
       # 'OTHER_CONSUMPTION', 'SOLAR_AGGREGATED', 'SOLAR_CONSUMPTION',
       # 'WASTE_AGGREGATED', 'WASTE_CONSUMPTION', 'WIND_OFFSHORE_AGGREGATED',
       # 'WIND_OFFSHORE_CONSUMPTION', 'WIND_ONSHORE_AGGREGATED',
       # 'WIND_ONSHORE_CONSUMPTION', 'CROSS_FLOW_DE_NL', 'CROSS_FLOW_NO_NL',
       # 'CROSS_FLOW_BE_NL', 'CROSS_FLOW_GB_NL', 'UPWARD_DISPATCH_y',
       # 'DOWNWARD_DISPATCH_y', 'TAKE_FROM_SYSTEM', 'FEED_INTO_SYSTEM',
       # 'Forecasted Load', 'GENERATION_FORECAST', 'SOLAR_FORECAST',
       # 'WIND_OFFSHORE_FORECAST', 'WIND_ONSHORE_FORECAST', 'PRICE_FORECAST',
       # 'Weekend', 'tempmax', 'tempmin', 'temp', 'feelslike', 'dew', 'humidity',
       # 'precip', 'precipprob', 'snow', 'windspeed', 'cloudcover',
       # 'solarradiation', 'solarenergy', 'uvindex', 'Total_Forecast_Error',
       #'Wind_Forecast_Error', 'Target(1)', 'Target(15)', 'Target(60)'
       # 'cluster'
    ]
feaacronymdict = {'IGCCCONTRIBUTION_DOWN': 'IGD', 'UPWARD_DISPATCH_x': 'UPD', 'DOWNWARD_DISPATCH_x': 'DD',
                  'Actual Load': 'AcL', 'FOSSIL_GAS_AGGREGATED' :'FAG', 'OTHER_AGGREGATED': 'OAG',
                  'CROSS_FLOW_NO_NL': 'CNO', 'CROSS_FLOW_BE_NL': 'CBE', 'CROSS_FLOW_GB_NL': 'CGB',
                  'UPWARD_DISPATCH_y': 'UPDy', 'TAKE_FROM_SYSTEM': 'TAKE',
                  'FEED_INTO_SYSTEM': 'FEED', 'PRICE_FORECAST': 'PFORE', 'Target': 'Targ',
                  'REGULATION_STATE_-1' : 'R-1', 'REGULATION_STATE_1': 'R1'
}
target_horizon = 'Target(1)'
train = createmodellingdf('train', '2018')
test = createmodellingdf('test', '2019')
model = xgb.XGBRegressor()#svm.SVR() #xgb.XGBRegressor()
C_mat = train.corr()
predictive_features = list(C_mat[abs(C_mat[target_horizon]) > 0.1].index)
l3 = [x for x in predictive_features if x not in ['cluster','Target(1)', 'Target(15)', 'Target(60)', 'Target_STANDARDIZED', 'MID_PRICE', 'MIN_PRICE', 'MAX_PRICE']]
C_mat2 = C_mat[l3].transpose()[target_horizon]
feaacronym = '_'.join([feaacronymdict[fea] for fea in l3])
# sns.heatmap(C_mat, vmax = .8, square = True)
# train[l3 + ['cluster']].hist(figsize = (12,10))
# plt.show()


clusterdfdict = dict()
X_train, X_test, y_train, y_test = tts2(train, test, l3, target_horizon)
reg = fit(model, X_train, y_train)
for cluster in train.cluster.unique():
    traincluster = train[train['cluster'] == cluster]
    testcluster = test[test['cluster'] == cluster]
    X_train, X_test, y_train, y_test = tts2(traincluster, testcluster, l3, target_horizon)
    y_pred = predict(reg, X_test, y_test)
    clusterdf = pd.DataFrame.from_dict({'timestamp': X_test.index,'prediction': y_pred, 'true': y_test,
                                        'error': np.subtract(y_pred, y_test),
                                        'cluster': [cluster]* len(y_test)}).set_index('timestamp')
    clusterdfdict[cluster] = clusterdf
errortablenonmeta = pd.concat(clusterdfdict.values()).sort_index()
rmse_by_cluster(errortablenonmeta)
print('rmsetotal non-meta: ',mean_squared_error(list(errortablenonmeta['true']), list(errortablenonmeta['prediction']), squared=False))

clusterdfdict = dict()
for cluster in train.cluster.unique():
    traincluster = train[train['cluster'] == cluster]
    testcluster = test[test['cluster'] == cluster]
    X_train, X_test, y_train, y_test = tts2(traincluster, testcluster, l3, target_horizon)
    reg = fit(model, X_train, y_train)
    y_pred = predict(reg, X_test, y_test)
    clusterdf = pd.DataFrame.from_dict({'timestamp': X_test.index,'prediction': y_pred, 'true': y_test,
                                        'error': np.subtract(y_pred, y_test),
                                        'cluster': [cluster]* len(y_test)}).set_index('timestamp')
    clusterdfdict[cluster] = clusterdf
errortablemeta = pd.concat(clusterdfdict.values()).sort_index()
rmse_by_cluster(errortablemeta)
print('rmsetotal meta: ',mean_squared_error(list(errortablemeta['true']), list(errortablemeta['prediction']), squared=False))


errortablemeta['nonmeta_prediction'] = list(errortablenonmeta['prediction'])
errortablemeta['nme'] = abs(errortablemeta['nonmeta_prediction']- errortablemeta['true'])
errortablemeta['me'] = abs(errortablemeta['prediction'] - errortablemeta['true'])
errortablemeta['closest'] = ['m: '
     if me < nme
     else 'nonmeta'
     for me, nme in zip(list(errortablemeta['me']), list(errortablemeta['nme']))]
errortablemeta['closer by'] = [str(round(nme -me,3))
     if me < nme
     else str(round(me -nme,3))
     for me, nme in zip(list(errortablemeta['me']), list(errortablemeta['nme']))]
errortablemeta =errortablemeta.sort_values(by=['error']).drop(['nme', 'me'], axis = 1)

etms = errortablemeta.sort_index()

os.makedirs('Results', exist_ok=True)
errortablemeta.to_pickle('Results/'+ feaacronym)

print(len(errortablemeta[errortablemeta['closest'] == 'm: '])/len(errortablemeta))
