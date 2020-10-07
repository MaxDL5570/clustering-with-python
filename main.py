import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Cust_Segmentation.csv')
df = df.drop('Address', axis=1)

x = df.values[:, 1:]
x = np.nan_to_num(x)

scaled_dataset = StandardScaler().fit_transform(x)

cluster_count = 3
k_means = KMeans(init='k-means++', n_clusters=cluster_count, n_init=12)
k_means.fit(x)

labels = k_means.labels_
df['Clus_km'] = labels

area = np.pi * (x[:, 1]) ** 2
plt.scatter(x[:, 0], x[:, 3], s=area, c=labels.astype(np.float), alpha=.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()
