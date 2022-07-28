import pandas as pd
feaacronymdict = {'IGCCCONTRIBUTION_DOWN': 'IGD', 'UPWARD_DISPATCH_x': 'UPD', 'DOWNWARD_DISPATCH_x': 'DD',
                  'Actual Load': 'AcL', 'FOSSIL_GAS_AGGREGATED' :'FAG', 'OTHER_AGGREGATED': 'OAG',
                  'CROSS_FLOW_NO_NL': 'CNO', 'CROSS_FLOW_BE_NL': 'CBE', 'CROSS_FLOW_GB_NL': 'CGB',
                  'UPWARD_DISPATCH_y': 'UPDy', 'TAKE_FROM_SYSTEM': 'TAKE',
                  'FEED_INTO_SYSTEM': 'FEED', 'PRICE_FORECAST': 'PFORE', 'Target': 'Targ',
                  'REGULATION_STATE_-1' : 'R-1', 'REGULATION_STATE_1': 'R1'
}
feas = 'Targ'
results = pd.read_pickle('Results/' + feas).sort_index()
features = [{v: k for k, v in feaacronymdict.items()}[fea] for fea in feas.split('_')]

results2 = pd.read_pickle('../Clustering/CleanData/Minutely/2019')[features]

results3 = pd.concat([results, results2], axis = 1)[['prediction', 'true', 'error', 'Target']]
etms = results3.sort_values(by=['error'])
results3.columns