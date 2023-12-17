#!/usr/bin/env python
# coding: utf-8

# # IRIS Flower Classification ML Project

# Import Libraries

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap


# Import the Dataset

# In[4]:


data=pd.read_csv('C:/Users/Data Analyst/Downloads/iris/Book2.csv')
data


# Extracting input Featuers and target Variables

# In[5]:


x=data[['sepal length','sepal width','petal length','petal width']].values
y=data['class'].values


# converting string values into numeric values using Encoder

# In[31]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)


# Splitting the Dataset into Training and Testing

# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Train a Logistic Regression Model

# In[35]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# Make a prediction on Testing Test

# In[10]:


y_pred=model.predict(x_test)


# Evaluate the Model

# In[34]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[12]:


sepal_length=x[:,0]
sepal_width=x[:,1]

class_labels=label_encoder.classes_
class_colors=['red','green','blue']
cmap=ListedColormap(class_colors)

plt.scatter(sepal_length,sepal_width,c=y,cmap=cmap)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Flower Classification Based on Sepal Measurements')

scatter_handles=[]
for class_label,class_color in zip(class_labels,class_colors):
  scatter_handles.append(plt.Line2D([],[],marker='o',markersize=8,linestyle='',color=class_color,label=class_label))
plt.legend(handles=scatter_handles)
plt.show()


# In[24]:


petal_length=x[:,2]
petal_width=x[:,3]

plt.scatter(petal_length,petal_width,c=y,cmap=cmap)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Iris Flower Classification Based on Petal Measurements')
scatter_handles=[]
for class_label,class_color in zip(class_labels,class_colors):
  scatter_handles.append(plt.Line2D([],[],marker='o',markersize=8,linestyle='',color=class_color,label=class_label))
plt.legend(handles=scatter_handles)
plt.show()


# Making a Prediction on New Data

# In[21]:


new_data=np.array([[5.1,3.5,1.4,0.2],[6.2,2.9,4.3,1.3],[7.3,2.8,6.5,2.3]])
new_prediction=model.predict(new_data)
print('New Data Predictions : ',prediction)
for i, prediction in enumerate(new_prediction):
  print(f'Data point {i+1}: {label_encoder.inverse_transform([prediction])[0]}')

