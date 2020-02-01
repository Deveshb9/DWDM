#!/usr/bin/env python
# coding: utf-8

# In[3]:


#library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as ptl 


# In[4]:


#dataset imports
data = pd.read_csv('C:/Users/Devesh Bhogre/Desktop/Programs/Python saves/capitalbikeshare-tripdata.csv')


# In[5]:


data.head()


# In[6]:


print(data.describe())


# In[7]:


#links
#https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
#iloc-> selects data by row number


# In[8]:


#features and class label seperation
 
x=data.iloc[:,[0,3,5]].values
y=data.iloc[:,-1].values
print(x[:5])
print(y[:5])


# In[9]:


#y needs encoding as it has categorical data
#
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y1 = le.fit_transform(y)
print("sample y:",y1[:5])
print("0 :",le.classes_[0])
print("1 :",le.classes_[1])


# In[10]:


#Splitting of Data into Training & Testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y1,test_size=0.25,random_state=0)  
print(y_test[:5])
print(x_test[:5])


# In[11]:


# Machine: Classifier | Classifier: Decision Tree
from sklearn.tree import DecisionTreeClassifier
#by default criterion=gini


# In[12]:


# Predicting the Test set results of entropy -> is the mesasure of randomness
# low entropy means high information gain and vice versa
cl_en = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=4, random_state=0)
cl_en.fit(x_train, y_train)
cl_en


# In[13]:


y_pred = cl_en.predict(x_test)
print("Predicting results of entropy: ")
print(y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred)) 
print ("Accuracy : \n",accuracy_score(y_test,y_pred)*100) 
print("Report : \n",classification_report(y_test, y_pred)) 


# In[14]:


cl_gini = DecisionTreeClassifier(criterion='gini', min_samples_leaf=4, random_state=0)
cl_gini.fit(x_train, y_train)
y_pred = cl_gini.predict(x_test)
print("Predicting results of entropy: ")
print(y_pred)
print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred)) 
print ("Accuracy : \n",accuracy_score(y_test,y_pred)*100) 
print("Report : \n",classification_report(y_test, y_pred)) 


# In[16]:


cl_gini.predict([[1012,31208,31108]])


# In[ ]:




