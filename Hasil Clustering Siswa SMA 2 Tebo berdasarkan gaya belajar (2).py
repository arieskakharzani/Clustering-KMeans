#!/usr/bin/env python
# coding: utf-8

# ## Persiapan Data

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[22]:


data = pd.read_csv(r'D:\Me\DAMIN\TB\dataset.csv')
data.head()


# In[23]:


import seaborn as sns
fitur = ['Visual', 'Auditorial', 'Kinestik']
data_fitur = data[fitur]

corr = data_fitur.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# ## Klastering Menggunakan K-Means

# In[24]:


X = data.iloc[:,1:3].values


# In[25]:


# Mengabaikan Warning
import warnings
warnings.filterwarnings('ignore')
#elbow method
wcss = []
for i in range(1,11):
    k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
#plot elbow curve
plt.plot(np.arange(1,11),wcss)
plt.xlabel('Clusters')
plt.ylabel('SSE')
plt.show()


# In[26]:


k_means_optimum = KMeans(n_clusters = 3, init = 'k-means++',  random_state=42)
y = k_means_optimum.fit_predict(X)
print(y)


# In[27]:


data['cluster'] = y  
# the above step adds extra column indicating the cluster number for each country


# In[28]:


data1 = data[data.cluster==0]
data2 = data[data.cluster==1]
data3 = data[data.cluster==2]


# In[31]:


kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)
kplot.plot3D(xline, yline, zline, 'black')
# Data for three-dimensional scattered points
kplot.scatter3D(data1.Visual, data1.Auditorial, data1.Kinestik, c='red', label = 'Kinestetik')
kplot.scatter3D(data2.Visual,data2.Auditorial,data2.Kinestik,c ='green', label = 'Auditorial')
kplot.scatter3D(data3.Visual,data3.Auditorial,data3.Kinestik,c ='blue', label = 'Visual')
plt.legend()
plt.title("Kmeans")
plt.show()


# ## Hasil Klastering

# In[30]:


# Membuat list dari anggota tiap clusternya
clusters_list = []
tipe = ['Kinestik', 'Auditorial', 'Visual']

for i in range(0, max(data['cluster'])+1):
    print('Cluster', tipe[i])
    print(data[data['cluster']==i]['Nama'].to_list())
    print("")

# Tampilkan list dari anggota tiap clusternya


# In[13]:


# 10 jumlah cluster
unique, counts = np.unique(y, return_counts=True)
cluster_counts = dict(zip(unique, counts))
print(cluster_counts)


# ## Klasifikasi Menggunakan Decission Tree

# In[32]:


data = data.drop(['Nama'],axis=1) 
data.head()


# In[15]:


x = data.drop(['cluster'],axis=1) 
y = data['cluster']


# In[16]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
import io


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train.values)
x_test = sc.transform(x_test.values)

x_test


# In[36]:


clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# # # Evaluasi Model

# In[40]:


print ('hasil akurasi :',metrics.accuracy_score(y_test,y_pred)*100,"%")


# In[39]:


plt.figure()
tree.plot_tree(clf.fit(x_train, y_train))


# ## Model Untuk Memprediksi Gaya Belajar Siswa

# In[1]:


visual = input("Masukkan nilai Visual: ")
auditorial = input("Masukkan nilai Auditorial: ")
kinestik = input("Masukkan nilai Kinestik: ")


listInput = []
listInput.extend([visual,auditorial,kinestik])
print(listInput)

deploy_df = pd.DataFrame (listInput).transpose()
deploy_df.columns = [['Visual','Auditorial','Kinestik']]

x_test_new = sc.transform(deploy_df)
y_pred = clf.predict(x_test_new)
print(y_pred[0])

# Hasil
if(y_pred[0]==0):
    print('Kamu masuk ke klater Kinestik')
elif(y_pred[0]==1):
    print('Kamu masuk ke klater Auditorial')
else:
    print('Kamu masuk ke klater Visual')


# In[ ]:




