#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import numpy as np


# In[4]:


from sklearn.cluster import KMeans


# In[6]:


X=np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])


# In[12]:


KMeans=KMeans(n_clusters=2,random_state=0).fit(X)


# In[16]:


KMeans.labels_


# In[17]:


KMeans.predict([[0,0],[4,4]])


# In[18]:


KMeans.cluster_centers_


# In[ ]:

#=================================================================================================#

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import numpy as np


# In[3]:


from sklearn.cluster import KMeans


# In[5]:


X=np.array([[5,3],[10,5],[15,12],[24,10],[24,10],[30,45],[85,70],[71,80],[60,78],[55,52],[80,91]])


# In[7]:


plt.scatter(X[:,0],X[:,1],label='True Position')


# In[12]:


KMeans=KMeans(n_clusters=2).fit(X)


# In[13]:


print(KMeans.cluster_centers_)


# In[14]:


print(KMeans.labels_)


# In[19]:


plt.scatter(X[:,0],X[:,1],c= KMeans.labels_,cmap='rainbow')


# In[23]:


plt.scatter(X[:,0],X[:,1],c= KMeans.labels_,cmap='rainbow')


# In[26]:


plt.scatter(KMeans.cluster_centers_[:,0],KMeans.cluster_centers_[:,1],color='black')


# In[ ]:







