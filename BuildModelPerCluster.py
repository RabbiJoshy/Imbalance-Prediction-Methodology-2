from SimpleUtils import *
import xgboost as xgb
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression
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
train = createmodellingdf('train', 'Train')
test = createmodellingdf('test', 'Test')
model = LinearRegression()#svm.SVR() #LinearRegression()#xgb.XGBRegressor()
def get_features(traindf, target_horizon, min_corr):
    C_mat = traindf.corr()
    predictive_features = list(C_mat[abs(C_mat[target_horizon]) > min_corr].index)
    l3 = [x for x in predictive_features if x not in ['cluster','Target(1)', 'Target(15)', 'Target(45)', 'Target(60)', 'Target_STANDARDIZED', 'MID_PRICE', 'MIN_PRICE', 'MAX_PRICE']]
    corr_features = C_mat[l3].transpose()[target_horizon]

    return l3
trainfeas = list(get_features(train, target_horizon, 0.2))
X_train, X_dev, y_train, y_dev = train_test_split(train[trainfeas + ['cluster']], train[[target_horizon,'cluster']], test_size=0.1, random_state=42)

def save_models(X_train, y_train, model):
    for cluster in X_train['cluster'].unique():
        X = X_train[X_train['cluster'] == cluster]
        y = y_train[y_train['cluster'] == cluster]
        print('fitting')
        model.fit(X.drop(['cluster'], axis = 1),y.drop(['cluster'], axis = 1))
        print('fitted')
        os.makedirs('models/', exist_ok=True)
        filename = 'models/' + str(cluster)
        pickle.dump(model, open(filename, 'wb'))
        print('model saved to', filename)

    return

save_models(X_train, y_train, model)

def create_predictionsdf(train, X_dev, y_dev):
    predictionsdictionary = {'True': list(y_dev['Target(1)']), 'Index' : y_dev.index}
    for cluster in train['cluster'].unique():
        # X = X_dev[X_dev['cluster'] == cluster]
        model = pickle.load(open('models/' + str(cluster), 'rb'))
        y_pred = model.predict(X_dev.drop(['cluster'], axis = 1))
        predictionsdictionary[str(cluster)+'_Prediction'] = [x for xs in y_pred for x in xs]
    predictions_df = pd.DataFrame.from_dict(predictionsdictionary).set_index('Index')

    return predictions_df

predictions_df_dev = create_predictionsdf(train, X_dev, y_dev)
predictions_df_test = create_predictionsdf(test, test[trainfeas + ['cluster']], test[[target_horizon,'cluster']])


predictions_df_dev['ClosestPredictor'] = predictions_df_dev.sub(predictions_df_dev['True'], axis=0).drop(['True'], axis =1).abs().idxmin(axis=1)
predictions_df_test['ClosestPredictor'] = predictions_df_test.sub(predictions_df_test['True'], axis=0).drop(['True'], axis =1).abs().idxmin(axis=1)

os.makedirs('PredictionData', exist_ok=True)
predictions_df_dev.to_pickle('PredictionData/Dev')
predictions_df_test.to_pickle('PredictionData/Test')