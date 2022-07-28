import os
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
reductionalgorithm = 'pca'
clusteringalgorithm = 'kmeans'
features = ['mean', 'max', 'min', 'std', 'skew', 'kurtosis']#, 'ac1', 'ac2']
def getdf(tt):
    train = pd.read_pickle('MetaFeatureData/' + tt[0])
    preclustertrain = pd.read_pickle('../Clustering/CleanData/' + tt[1])[['Target(1)']]
    train['Target(1)'] = list(preclustertrain['Target(1)'])
    train = train.fillna(0)
    return train
train = getdf(['train30','Train'])
test = getdf(['test30','Test'])
##############################################################################################################################
# # train.corr()
# # sns.heatmap(train.corr())
#
if reductionalgorithm == 'TSNE':
    train = train.sample(n=200000)
    P = TSNE(n_components=2, perplexity= 50, verbose = 1)
    reduced = P.fit_transform(train.drop(['Target', 'Target(1)'], axis=1))
elif reductionalgorithm == 'pca':
    P = PCA(n_components=2)
    P.fit(train.drop(['Target', 'Target(1)'], axis=1)[features])
    reduced = P.transform(train.drop(['Target', 'Target(1)'],axis =1)[features])
    reducedtest = P.transform(test.drop(['Target', 'Target(1)'],axis =1)[features])

df = pd.DataFrame(reduced, columns = ['Dimension1', 'Dimension2'])
if reductionalgorithm == 'pca':
    dftest = pd.DataFrame(reducedtest, columns = ['Dimension1', 'Dimension2'])
    test['Dimension1'] = list(dftest['Dimension1'])
    test['Dimension2'] = list(dftest['Dimension2'])
train['Dimension1'] = list(df['Dimension1'])
train['Dimension2'] = list(df['Dimension2'])

os.makedirs('ReducedData/' + reductionalgorithm + '/', exist_ok=True)
train.to_pickle('ReducedData/' + reductionalgorithm + '/' + 'train')
test.to_pickle('ReducedData/' + reductionalgorithm + '/' + 'test')

##############################################################################################################################

train = pd.read_pickle('ReducedData/' + reductionalgorithm + '/' +'train')
test = pd.read_pickle('ReducedData/' + reductionalgorithm + '/' +'test')

if clusteringalgorithm == 'kmeans':
    kmeansfeatures =  ['Target']
    # plt.switch_backend('Agg')
    # model = KElbowVisualizer(KMeans(), k=(3, 8), metric = 'silhouette')
    # model.fit(train[['Dimension1', 'Dimension2']])
    # kmeans = KMeans(n_clusters=5, random_state=0).fit(train[['Target']])
    kmeans = KMeans(n_clusters=5, random_state=0).fit(train[['Dimension1', 'Dimension2']])
    train['cluster'] = kmeans.labels_
    trainvis = train.sample(n =10000)
    sns.scatterplot(data=trainvis, x= trainvis.index, y='Target', hue=trainvis.cluster)
    #sns.scatterplot(data=trainvis, x='Dimension1', y= 'Dimension2',hue=kmeans.labels_)

    # test['cluster'] = kmeans.predict(test[features])
    testvis =test.sample(n=1000)
    test['cluster'] = kmeans.predict(test[['Dimension1', 'Dimension2']])
    sns.scatterplot(data=testvis, x='Dimension1', y='Dimension2', hue='cluster')
    sns.scatterplot(data=testvis, x=testvis.index, y='Target', hue='cluster')
if clusteringalgorithm == 'HDBSCAN':
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5000, min_samples=400).fit(train[['Dimension1', 'Dimension2']])
    # clusterer.condensed_tree_.plot(select_clusters=True)
    color_palette = sns.color_palette('Paired', 15)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
    plt.scatter(train['Dimension1'], train['Dimension2'], s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)

    test_labels, strengths = hdbscan.approximate_predict(clusterer, test_points)
    train['cluster'] = clusterer.labels_
    test['cluster'] = clusterer.test_labels
    # view = train.sample(n=100)
    # sns.scatterplot(data=view, x=view.index, y= 'Target',hue='cluster')

os.makedirs('ClusteredData/' + reductionalgorithm, exist_ok=True)
train.to_pickle('ClusteredData/' + reductionalgorithm + '/' + 'train')
test.to_pickle('ClusteredData/' + reductionalgorithm + '/' + 'test')

##############################################################################################################################

import pandas as pd
import seaborn as sns
reductionalgorithm = 'pca'
train = pd.read_pickle('ClusteredData/' + reductionalgorithm + '/' + 'train')
test = pd.read_pickle('ClusteredData/' + reductionalgorithm + '/' + 'test')
train['set'] = ['train'] * len(train)
test['set'] = ['test'] * len(test)
total = pd.concat([train, test])
ax = sns.boxplot(y="Target(1)", x="cluster", data=total, hue = 'set')
ax = sns.boxplot(y="ac2", x="cluster", data=train)
ax = sns.boxplot(y="Target(1)", x="cluster", data=test)

train['cluster'].value_counts()
test['cluster'].value_counts()
train['Target'].mean()
test['Target'].mean()