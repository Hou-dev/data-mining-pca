#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import numpy as np
from scipy.stats import zscore


# In[2]:


# reading data to dataframe
data = pd.read_excel('Exam_Scores.xlsx')
data
score = data[['Exam1','Exam3']]


# In[3]:


score['Exam1'] = zscore(score['Exam1'])
score['Exam3'] = zscore(score['Exam3'])
score


# In[4]:


# finding the mean of x and y
mean_x = np.mean(score['Exam1'])
mean_y = np.mean(score['Exam3'])
#mean_vec = np.array([[mean_x],[mean_y]])
print(mean_y)


# In[5]:


# caculate the covariance matrix
cov_mat = np.cov([score['Exam1'],score['Exam3']])
print(cov_mat)


# In[6]:


# find the the eigenvalues and eigenvectors using numpy
e_val, e_vec = np.linalg.eig(cov_mat)
print('Value')
print(e_val)
print('')
print('Vector')
print(e_vec)


# In[7]:


# sorting the eigenvalue
eig_pairs = [(np.abs(e_val[i]), e_vec[:,i]) for i in range(len(e_val))]
eig_pairs.sort(key=lambda x:x[0],reverse=True)
for i in eig_pairs:
    print(i[0])


# In[8]:


# finding the covariance of the eigenpairs
matrix = np.hstack((eig_pairs[0][1].reshape(2,1), eig_pairs[1][1].reshape(2,1)))
print('Matrix:\n', matrix)


# In[9]:


# Put scores into a matrix and rehape to mutiply by eigenpairs to find PCA
score_mat = score.as_matrix().reshape(2,12)

transformed = matrix.T.dot(score_mat)
assert transformed.shape == (2,12)
# print the transformed matrix
print(transformed)
a1 = transformed[0]
b1 =  transformed[1]
c1 = [a1,b1]


# In[10]:


# sum squared distance and variation of PCA1
from sklearn.metrics.pairwise import euclidean_distances
var1 = euclidean_distances(c1)/(len(a1)-1)
print(var1)


# In[12]:


# from the slides we find PCA2 by mutiplying the -0.41*E1 and 0.89*E2
#a[0].dot(transformed[0])
a = np.array(-0.41*transformed[0])
b = np.array(0.89*transformed[1])


# In[13]:


# sum squared distance and variation of PCA2
a.reshape(-1,1)
b.reshape(-1,1)
c = [a,b]
var2 = euclidean_distances(c)/(len(a)-1)
print(var2)


# In[16]:


# making screeplot using the variation
import matplotlib.pyplot as plt
objects = ('PCA1','PCA2')
pos = np.arange(len(objects))
ver = [0.51,0.204]

plt.bar(pos, ver, align='center', alpha=0.5)
plt.xticks(pos, objects)
plt.ylabel('Variation')
plt.title('PCA and Variation')

plt.show()


# In[17]:


# Below is the PCA implement using only scipy and numpy


# In[18]:


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
sklearn_transf = sklearn_pca.fit_transform(score.T)


# In[19]:


# find the PCA1 and PCA2 using sklearn
from sklearn.decomposition import PCA
pca1 = PCA(1).fit(score)
# printing the percentage of variance for PCA1
print('Percentage Explained: ', pca1.explained_variance_ratio_*100)


# In[20]:


pca2 = PCA(2).fit(score)
#printing the perccentage variance for PCA2
print('Percentage Explained: ',pca2.explained_variance_ratio_*100)


# In[21]:


# printing the screeplot
import matplotlib.pyplot as plt
pca = PCA().fit(score)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()


# In[ ]:




