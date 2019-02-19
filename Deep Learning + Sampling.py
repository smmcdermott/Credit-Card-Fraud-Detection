#!/usr/bin/env python
# coding: utf-8

# ## Importing the Libraries and Dataset

# In[1]:


import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import seaborn as sn

np.random.seed(2)


# In[2]:


# Load the dataset
data = pd.read_csv('creditcard.csv')


# ## Data exploration

# In[3]:


data.head()


# ## Display the Histograms

# In[4]:


dataset2 = data
# Fit the screen
fig = plt.figure(figsize=(15, 12))
# Set the main title
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
# For each column in the dataset
for i in range(dataset2.shape[1]):
    # Get 6 rows with 3 plots per row
    plt.subplot(8, 4, i+1)
    # gca=get current axis
    f = plt.gca()
    # Set the title of each plot to its column name
    f.set_title(dataset2.columns.values[i])

    # Get the number of unique values as the number of bins
    vals = np.size(dataset2.iloc[:, i].unique())
    # If there are more than 10 unique values, then it defaults to 100. Ensures that it will not crash my system
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# ## Correlation between Explanatory Variables

# In[5]:


# Get the correlation of the dataset witht the e_signed column
# Plot those results on a bar chart
"""Parameters-figsize-shape of the plot
              title-title of plot
              fontsize-size of the font
              rot-45 degree for x-axis labels
              grid-plot on gridlines"""
dataset2 = dataset2.drop(columns = ["Class"])
dataset2.corrwith(data["Class"]).plot.bar(
        figsize = (20, 10), title = "Correlation with Explanatory Variables", fontsize = 15,
        rot = 45, grid = True)


# ## Correlation Matrix

# In[6]:


# Set to a white background
sn.set(style="white")

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ## Pre-processing

# In[7]:


from sklearn.preprocessing import StandardScaler
# Apply Standard Scaler to the amount data and store that in a new column called normalized amount
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
# Drop the original amount column since we no longer need it
data = data.drop(['Amount'],axis=1)


# In[8]:


# Explore the data
data.head()


# In[9]:


# Drop the time column since it is irrelevant to our model
data = data.drop(['Time'],axis=1)
data.head()


# In[10]:


# Set our feature data to all features that is not the the dependent variable (class)
X = data.iloc[:, data.columns != 'Class']
# Set Class as our dependent variable
y = data.iloc[:, data.columns == 'Class']


# In[11]:


# Explore our dependent variable
y.head()


# In[12]:


# Split the dataset into Training and Testing Data with a 70:30 Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)


# In[13]:


# View the shape of the dataset (200,000 rows with 29 columns)
X_train.shape


# In[14]:


# View the shape of the test set
X_test.shape


# In[15]:


# Create np arrays so that it will work in our deep learning model
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ## Deep neural network

# In[16]:


# Import the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[17]:


# Create a sequential model with many dense layers
# Drop out in the middle to avoid overfitting
# Then create more dense layers with relu activation functions
# Use sigmoid activation function at the end since we have a binary target variable
model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])


# In[18]:


model.summary()


# ## Training

# In[19]:


# Compile the model with the adam optimizer with binary cross_entropy
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Fit the model to the training data with a batch size of 15
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[20]:


# Get the score by evaluating our model on the test data
score = model.evaluate(X_test, y_test)


# In[21]:


# Display the score
# Results-99.93 % accuracy
print(score)


# ## Plot the Confusion Matrix

# In[22]:


import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[23]:


# Get the predicted values from the model
y_pred = model.predict(X_test)
# Get the actual values
y_test = pd.DataFrame(y_test)


# In[24]:


# Get the confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred.round())


# In[25]:


print(cnf_matrix)


# In[26]:


plot_confusion_matrix(cnf_matrix, classes=[0,1])


# In[27]:


plt.show()


# In[28]:


# Get the predicted values based on the training data
y_pred = model.predict(X)
# Get the expected values based on the y_data
y_expected = pd.DataFrame(y)
# Show confusion matrix based on the training data
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# ## Undersampling

# In[29]:


# Get the number of fraudulent transactions
fraud_indices = np.array(data[data.Class == 1].index)
number_records_fraud = len(fraud_indices)
print(number_records_fraud)


# In[30]:


# Get the non-fradulent transactions
normal_indices = data[data.Class == 0].index


# In[31]:


# Get the same number of non-fradulent transactions as fradulent transactions
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
print(len(random_normal_indices))


# In[32]:


# Concatenate the indexes to combine all the fraudlent and non_fraudulent transactions
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
print(len(under_sample_indices))


# In[33]:


under_sample_data = data.iloc[under_sample_indices,:]


# In[34]:


# Get the X samples as all features that arent Class
X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
# Set Class as dependent variable
y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']


# In[35]:


# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample, test_size=0.3)


# In[36]:


# Convert the data into np.arrays so it works in the model
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[37]:


# View the shape of the data
model.summary()


# In[38]:


# Compile the model to train the model again on the undersampled data
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[39]:


# Create the confusion matrix on the undersampled data
y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[40]:


# Predict the model on the whole data set
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# ## SMOTE

# In[41]:


get_ipython().run_cell_magic('bash', '', '#pip install -U imbalanced-learn')


# In[42]:


# Import smote
from imblearn.over_sampling import SMOTE


# In[ ]:


# Get resampled data by fitting smote to the dataset
X_resample, y_resample = SMOTE().fit_sample(X,y.values.ravel())


# In[ ]:


# Convert them to dataframes
y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)


# In[ ]:


# Split the data into training and sets
X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)


# In[ ]:


# Convert to numpy arrays to work in the model
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[ ]:


# Train the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[ ]:


# Create the confusion matrix on the test data
y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[ ]:


# Create the confusion matrix on the entire dataset
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[ ]:




