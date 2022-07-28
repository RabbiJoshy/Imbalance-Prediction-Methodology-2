from MetaFeatureUtils import *
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

argv = [30, ['mean', 'std', 'kurtosis', 'max','skew', 'min', 'ac1', 'ac2'], 'ignore', 'Train'] #ac1,ac2
train = pd.read_pickle('/Users/joshuathomas/Desktop/Thesis/Thesis Large Files/WindowData/' + argv[3])
train['window'] = list(train['window_' + str(argv[0])])
train[argv[2]] = [0]* len(train)

#MetaDFs
trainmetadf = create_metafeaturedf(train, argv[1], argv[0])
expandedmetadf = expand_metafeaturedf(trainmetadf)
cleanmeta_train = normalise(expandedmetadf, argv[2])
cleanmeta_train['Target'] = list(train['Target'])
cleanmeta_train = cleanmeta_train.set_index(train.index).sort_index()
# finaldf  = cleanmeta_train.drop([argv[2]], axis = 1)
finaldf = cleanmeta_train
finaldf.to_pickle('MetaFeatureData/' + argv[3] + str(argv[0]))

