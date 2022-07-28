import pandas as pd
from sklearn.metrics import mean_squared_error

def createmodellingdf(tt, ttcaps):
    preclustertrain = pd.read_pickle('../Clustering/CleanData/' + ttcaps)
    train_clustered = pd.read_pickle('ClusteredData/pca/' + tt)  # .sample(n=200000)
    train = preclustertrain[preclustertrain.index.isin(train_clustered.index)]
    train['cluster'] = list(train_clustered['cluster'])

    return train
def tts2(traindf, testdf, features, target_horizon):
    X_train = traindf[features]
    X_test = testdf[features]
    y_train = list(traindf[target_horizon])
    y_test = list(testdf[target_horizon])

    return X_train, X_test, y_train, y_test
def tts(df, features, target_horizon):
    X = df[features]
    y = list(df[target_horizon])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test
def fit(model, X_train, y_train):
    reg = model
    print('fitting')
    reg.fit(X_train, y_train)
    return reg
def predict(reg, X_test, y_test):
    print('predicting')
    y_pred = reg.predict(X_test)
    print('predicted')
    # rmse = mean_squared_error(y_test,y_pred, squared=False)
    # # print(rmse)
    # data_tuples = list(zip(y_test,y_pred, y_test - y_pred))
    # errors = pd.DataFrame(data_tuples, columns=['true','prediction', 'error'])
    #train['error'] = np.subtract(y_test,y_pred)

    return y_pred #rmse, errors
def rmse_by_cluster(errortable):
    for cluster in errortable['cluster'].unique():
        clustererror = errortable[errortable['cluster'] == cluster]
        rmse = mean_squared_error(list(clustererror['true']), list(clustererror['prediction']), squared=False)
        print(cluster, ':' ,rmse)

    return