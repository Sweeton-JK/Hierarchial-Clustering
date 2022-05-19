#Importing Libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#Read CSV

iris_data=pd.read_csv("C:\\Users\\SWEETON\\Desktop\\iris.csv",index_col=0)
x=iris_data.iloc[:, :-1]
y=iris_data.iloc[:,:-1]
iris_data=iris_data.iloc[:, :-1]
iris_data.head(10)

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='single')
y_hc=hc.fit_predict(iris_data)
a=iris_data.iloc[:,0]
b=iris_data.iloc[:,1]
plt.scatter(a,b)

dendrogram=sch.dendrogram(sch.linkage(x,method='single'))

hc = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(iris_data)
a=iris_data.iloc[:,0]
b=iris_data.iloc[:,2]
plt.scatter(a,b,color='green')

dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')
y_hc=hc.fit_predict(iris_data)
a=iris_data.iloc[:,1]
b=iris_data.iloc[:,2]
plt.scatter(a,b,color='red')

dendrogram=sch.dendrogram(sch.linkage(x,method='complete'))