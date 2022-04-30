#!/usr/bin/env python
# coding: utf-8

# # Kernel SVM

# ## Importing the libraries

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# ## Importing the dataset

# In[5]:


dataset = pd.read_csv('datasets/cardio_preprocessed.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ## Feature Scaling

# In[7]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Training the Kernel SVM model on the Training set

# In[8]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# ## Making the Confusion Matrix

# In[9]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[10]:


pickle.dump(classifier, open('deploy/cardio_kernel_svm.pkl' ,'wb'))


# In[ ]:





# In[ ]:




