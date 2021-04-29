#!/usr/bin/env python
# coding: utf-8

# In[8]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[37]:


df = web.DataReader('OCGN', data_source='yahoo', start='2020-01-01', end='2021-04-28')


# In[38]:


df


# In[39]:


# Get Number of Rows and Columns in the Data Set.
df.shape


# In[40]:


# Creating a new dataframe with only the Closing price at the end of the day.
data = df.filter(['Close'])

# Convert dataframe to a numpy array.
dataset = data.values

#get the number of rows to train the model on.
training_data_len = math.ceil(len(dataset)* .8)

training_data_len


# In[41]:


# Scale the data to improve the model using the Min and Max Scaler method.
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# View scaled data
scaled_data


# In[42]:


# Creating the training data set.
# Create the scaled training data set.
train_data = scaled_data[0:training_data_len, :]

# Splitting the data into x_train and y_train data sets.

# x_train is the dependent variable.
x_train = []

# y_train is the independent variable.
y_train = []

# Appending x_train to the first 60 set of values in the data (0 to 59) and y_train to the rest which starts at position 61.
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()


# In[43]:


# Convert the x_train and y_train to numpy arrays to be used for training the LSTM model.
x_train, y_train = np.array(x_train), np.array(y_train)


# In[44]:


# Reshaping the x_train data set. LSTM model expects the data to be 3 Dimensional and it's currently 2 Dimensional so we have to convert it to 3 Dimensions.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[45]:


# Build the LSTM Model.
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
# Models architecture with 25 neurons.
model.add(Dense(25))

# Models architecture with 1 neuron.
model.add(Dense(1))


# In[46]:


# Compiling the model.

# Used to improve upon the loss function. The loss function is used to measure how well the model did upon training.
model.compile(optimizer='adam', loss='mean_squared_error')


# In[47]:


# Train the model with our data set.

# epochs is the number of iterations for a forward and backward neural network.
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[48]:


# Creating the testing data set.
# Creating a new array containing scaled values from index 01/01/2021 to 04/29/2021.
test_data = scaled_data[training_data_len - 60:, :]

# Create the data sets x_test and y_test.
x_test = []

# All of the values we want our data to predict.
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[49]:


# Converting the data into a numpy array to use in the LMSTO.
x_test = np.array(x_test)


# In[50]:


# Reshape the data. x_test.shape[0] is the number of rows. x_test.shape[1] is the number of columns.
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[51]:


# Get the models predicted price values. We want the predictions to contain the same values as the y_test data set. 
# Predictions are made off of the x_test data set.
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[52]:


# Get the root mean squared error (RMSE). Measures how accurate the model predicts the response. The lower values mean it's a better fit.
rmse = np.sqrt( np.mean( predictions - y_test )**2 )
rmse

# Note we got a 1.995, which is really good here. A value of 0 means the predictions were perfect so here it's close.


# In[56]:


# Plot the data so we can visualize it.
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the models data.
plt.figure(figsize=(16,8))
plt.title('OCGN Algorithm Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])

# Displays the predicted price
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[57]:


# Showing the actual price and predicted prices.
valid


# In[63]:


# Getting the quote.
ocgn_quote = web.DataReader('OCGN', data_source='yahoo', start='2020-01-01', end='2021-04-27')

# Creating a new dataframe.
new_df = ocgn_quote.filter(['Close'])

# Retrieve the last 60 day closing price values and convert the dataframe into an array.
last_60_days = new_df[-60:].values

# Scale the data to be values between 0 and 1. Not using fit transform because we want to use the same values as above.
last_60_days_scaled = scaler.transform(last_60_days)

# Creating an empty list.
X_test = []

# Append the last 60 days to the X_test list that were scaled to the data set.
X_test.append(last_60_days_scaled)

# Conver the X_test data set to a numpy array.
X_test = np.array(X_test)

# Reshape the data.
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the predicted scale price.
pred_price = model.predict(X_test)

# Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[64]:


# Getting the quote.
ocgn2 = web.DataReader('OCGN', data_source='yahoo', start='2020-04-28', end='2021-04-28')
print(ocgn2['Close'])


# In[12]:


# Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date, fontsize=18')
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

