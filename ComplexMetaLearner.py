import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
predictions_df_dev = pd.read_pickle('PredictionData/Dev')
predictions_df_test = pd.read_pickle('PredictionData/Test')#.sort_values(by = 'ClosestPredictor')
metafeaturetrain = pd.read_pickle('MetaFeatureData/Train30').drop(['Target', 'ac1', 'ac2'], axis =1).loc[predictions_df_dev.index, :]
metafeaturetest = pd.read_pickle('MetaFeatureData/Test30').drop(['Target'], axis =1).fillna(0)

#Fit Meta-Learner
# c_mat = metafeaturetest.corr()
clf = LogisticRegression(random_state=0)
clf.fit(metafeaturetrain, predictions_df_dev['ClosestPredictor'])
best_test_predictions = clf.predict(metafeaturetest)

#Compare Meta-Learner to Theoretical best
best_test_true = list(predictions_df_test['ClosestPredictor'])
print(accuracy_score(best_test_true, best_test_predictions))
cm = pd.DataFrame(confusion_matrix(best_test_true, best_test_predictions))

#Evaluate Actual Meta-Learner
meta_predictions = [predictions_df_test[b][i] for i,b in enumerate(best_test_predictions)]
print(mean_squared_error(list(predictions_df_test['True']), meta_predictions))
print(mean_absolute_error(list(predictions_df_test['True']), meta_predictions))

#Evaluate Theoretical Best
predictions_df_test['BestPrediction'] = [predictions_df_test[predictions_df_test['ClosestPredictor'][i]][i] for i in range(len(predictions_df_test))]
print(mean_squared_error(list(predictions_df_test['BestPrediction']), list(predictions_df_test['True']), squared = False))
print(mean_absolute_error(list(predictions_df_test['BestPrediction']), list(predictions_df_test['True'])))
view = predictions_df_test[['True', 'BestPrediction']]
view['Error'] = view['True'] - view['BestPrediction']

#Evaluate Non-Theoretical


a = view['True'].sort_index()
b = a.shift(1).fillna(0)
a = list(a)
b = list(b)
mean_absolute_error(a,b)
mean_squared_error(a,b, squared = False)

view2 = predictions_df_test.sort_values(by = 'BestPrediction')

counter = 0
for i in range(len(a)):
    if a[i] == b[i]:
        counter+=1
print(counter/len(a))

import numpy as np
c = list(map(abs, np.subtract(np.array(a), np.array(b))))
np.mean(c)
