#!/usr/bin/env python
# coding: utf-8

# # LGM - Data Science Internship December 2023
# 
# # Task-3 Prediction using Decision Tree  Algorithm 

# # Importing the Libraries

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Import the Dataset

# In[7]:


dataset = pd.read_csv("C:/Users/Data Analyst/Downloads/iris.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# # Splitting the Dataset into the Training set and Test set

# In[9]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[11]:


print(X_train)


# In[13]:


print(y_train)


# In[14]:


print(X_test)


# In[15]:


print(y_test)


# # Feature Scaling

# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[19]:


print(X_train)


# In[20]:


print(X_test)


# # Training the Decision Tree Classification Model on the Training set  

# In[25]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) 
classifier.fit(X_train, y_train)


# # Predicting the Test Result set

# In[27]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# # Making the Confusion Matrix

# In[30]:


from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# Accuracy is 100 %

# # Visualising things

# In[33]:


import seaborn as sns


# In[36]:


dataset


# In[37]:


sns.pairplot(data=dataset , hue='Species')
plt.show()


# In[45]:


col = dataset.columns[:-1]
classes = dataset['Species'].unique().tolist()
from sklearn.tree import plot_tree
plt.figure(figsize=(16,10))
plot_tree(classifier, feature_names=col, class_names=classes,filled=True)

