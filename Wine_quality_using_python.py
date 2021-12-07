
import pandas as pd

data= pd.read_csv("C:/Users/prudviraj.e/Downloads/wine.csv")
print(data.describe())
print(data.head())
print(data.columns)
print(data.iloc[:,1:].describe())

# PCA
from sklearn.decomposition import PCA
df=data.iloc[:,1:]

from sklearn.preprocessing import minmax_scale
scaled_df=minmax_scale(df)
scaled_df=pd.DataFrame(scaled_df)
cor=scaled_df.corr()
pca_scores=PCA().fit_transform(scaled_df)
pca_scores=pd.DataFrame(pca_scores)
five_pc_scores=pca_scores.iloc[:,0:5]

#clustering(As told 7 clusters)
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("Dendogram")
dend=sch.dendrogram(sch.linkage(five_pc_scores,method='ward'))

#cluterring with give n_cluster value
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='ward')
clusters=cluster.fit_predict(five_pc_scores)
clusters=pd.DataFrame(clusters)
clusters.columns=['cluster_no']

new_data=pd.concat([five_pc_scores,clusters],axis=1)
#only taking third cluster data
new_data=pd.concat([new_data,data['Alcohol']],axis=1)
third_cluster_data=new_data[new_data['cluster_no']==2]


#Linear regression
from sklearn.model_selection import train_test_split
third_cluster_data.columns
X_train, X_test, y_train, y_test = train_test_split(third_cluster_data.loc[:,~third_cluster_data.columns.isin(['Alcohol','cluster_no'])],third_cluster_data['Alcohol'],random_state=33)

from sklearn.linear_model import LinearRegression
lm=LinearRegression(normalize=True)

m1=lm.fit(X_train,y_train)

from statsmodels.api import OLS
OLS(y_train,X_train).fit().summary()


