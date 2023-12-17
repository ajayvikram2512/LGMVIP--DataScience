#!/usr/bin/env python
# coding: utf-8

# # LGM - Data Science Internship - Dec 2023
# 
# 
# 
# # Stock Market Prediction And Forecasting Using Stacked LSTM

# In[61]:


# Importing the necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# # Data Processing

# In[8]:


data = pd.read_csv('C:/Users/Data Analyst/OneDrive/Desktop/task-2.csv')


# In[9]:


data.head() # Shows first 5 rows of Dataset


# In[10]:


data.tail() # Shows last 5 rows of dataset


# In[12]:


#Sort the dataset according to the date
data['Date'] = pd.to_datetime(data['Date']) 
df = data.sort_values(by='Date') 
df.head()


# # Index is not in order, Lets make it in order

# In[13]:


df.reset_index(inplace=True)


# In[14]:


df.head()


# # Plotting the graph of stocks at close

# In[16]:


plt.plot(df['Close'])


# In[17]:


df1 = df['Close']


# # Feature Scaling

# In[18]:


scaler = MinMaxScaler (feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1)) 
df1


# # Splitting the data into training and test sets 

# In[26]:


training_size = int(len(df1)*0.70) 
test_size = len(df1)-training_size

train_data = df1[0:training_size,:] 
test_data = df1[training_size:len (df1), :1]


# In[27]:


training_size , test_size


# In[39]:


#convert an array of values into a dataset matrix
def datasetCrtd(dataset , time_step=1):
   datax, datay = [], []
   for i in range(len(dataset)-time_step-1):
       a = dataset[i:(i+time_step), 0]
       datax.append(a)
       datay.append(dataset[i + time_step, 0])
   return np.array(datax), np.array(datay)


# In[40]:


time_step = 100
X_train, y_train = datasetCrtd(train_data, time_step) 
X_test, ytest = datasetCrtd(test_data, time_step)


# In[42]:


print(X_train.shape), print(y_train.shape)


# In[46]:


#Reshape input which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# # Building the Model

# In[6]:


pip install tensorflow


# In[47]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[51]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1))) 
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
Model: "sequential"


# In[55]:


model.fit(X_train, y_train, validation_split=0.1, epochs=60, batch_size=64, verbose=1)


# In[56]:


# Prediction of scaled test set
test_predict = model.predict(X_test)


# In[57]:


# Transform to original test set
test_predict1 = scaler.inverse_transform(test_predict)


# In[58]:


test_predict1


# # Performance Metrics
# 
# # Mean Square error

# In[62]:


import math
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
math.sqrt(mean_squared_error(ytest, test_predict)) #MSE is Low


# Conclusion : Hence, We Can Conclude that the mean squared error is low and is 0.032445
