#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd     #pandas helps in reading the file and converting it into dataframe
import numpy as np      #numpy helps in converting dataframe into numpy arrays


# In[2]:


train=pd.read_csv("F:\MACHINE LEARNING\Project/finance_train_data.csv")  #reading the given csv file


# In[3]:


train.head()      #it will display top 5 elements of the file


# In[4]:


train.tail()       #it will display bottom five elements of the file


# In[5]:


train.describe()        #it will return statsitical analysis of the dataframe


# In[6]:


train.dtypes          #it will return datatype of each column


# In[7]:


train.info()       #it will return number of columns, total number of elements, column names, datatype of each column and number of non-null count of individual columns.


# In[8]:


train.columns     #it will return column names.


# In[9]:


train["Loan_Status"].value_counts()      


# In[10]:


# Using the below piece of code we are checking how many null(NaN) elements are present in each row.
category=["Gender","Married",'Dependents','Loan_Amount_Term','Credit_History','Self_Employed','LoanAmount']
for i in category:
    print(train[i].isnull().value_counts())


# In[11]:


# using the below piece of code we are filling the NaN with mode,mean.
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)     #mode-the element which occurs many times.
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)  #mean-the average of all elements 


# In[12]:


#Using this small piece of code we are converting non-integer values into integers.
from sklearn.preprocessing import LabelEncoder
category= ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status'] 
encoder= LabelEncoder()
for i in category:   
    train[i] = encoder.fit_transform(train[i]) 
train.dtypes


# In[13]:


train.corr()          #corr() method will return how one variable is dependent with other one. A number closer to 1 means highly correlated/dependent and vice-versa. 


# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array

# In[14]:


X=train[['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']].values
X[0:5]    #we are converting dataframe into numpy arrays.


# In[15]:


Y=train["Loan_Status"].values      #we are converting dataframe into numpy arrays.
Y[0:5]


# Data Standardization give data zero mean and unit variance, it is good practice

# In[16]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[17]:


#we are splitting the train dataframe into train test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # Logistic Regression model 

# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)    #Building/Training the model.
LR


# In[19]:


yhat = LR.predict(X_test)          # making predictions based on the trained model.
yhat


# In[20]:


yhat_prob = LR.predict_proba(X_test)     #it willreturn the probability of either being 0 or 1.
yhat_prob


# In[21]:


import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
accuracy = metrics.accuracy_score(yhat,y_test)
print("Accuracy : %s" % "{0:.3%}".format(accuracy))     #Accuracy of the model
print("r2_score:",r2_score(y_test, yhat))               #R2 value
print("mse:",mean_squared_error(y_test, yhat))          #MSE(Mean square Error) of the model


# # KNN(K-th Nearest Neighbour Model)

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sklearn.metrics as metrics


# In[40]:


#Finding the best value of K
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[24]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# In[25]:


k = 7
#Train Model  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)     #Model Training
neigh


# In[27]:


yhat = neigh.predict(X_test)                    #Predicting values by using trained model
yhat


# In[28]:


import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
accuracy = metrics.accuracy_score(yhat,y_test)
print("Accuracy : %s" % "{0:.3%}".format(accuracy))            #Accuracy of the model
print("r2_score:",r2_score(y_test, yhat))                      #R2 value
print("mse:",mean_squared_error(y_test, yhat))                 #MSE(Mean Square Error) of the model


# # For the given test dataset we need to predict LoanStatus using given independent variables and need to update the csv file

# In[29]:


df=pd.read_csv("F:\MACHINE LEARNING\Project/finance_test_data.csv")     #reading the test dataset for which we need to predict loanStatus.


# In[30]:


#Data PreProcessing(Filling NaN with mean,mode)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


# In[31]:


#converting non-integer datatypes into integers
from sklearn.preprocessing import LabelEncoder
category= ['Gender','Married','Dependents','Education','Self_Employed','Property_Area'] 
encoder= LabelEncoder()
for i in category:   
    df[i] = encoder.fit_transform(df[i]) 
df.dtypes


# In[32]:


#converting dataframe into numpy arrays
N=df[['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']].values


# In[33]:


from sklearn import preprocessing
N = preprocessing.StandardScaler().fit(N).transform(N.astype(float))


# In[34]:


yhat = neigh.predict(N)        #predicting values using KNN Model


# In[35]:


print(yhat)


# In[36]:


df["Loan_Status"]=yhat


# In[37]:


df.head()


# In[38]:


df["Loan_Status"].replace({1:"Y",0:"N"},inplace=True)
df["Gender"].replace({1:"Male",0:"Female"},inplace=True)
df["Married"].replace({1:"Yes",0:"No"},inplace=True)
df["Education"].replace({1:"Not Graduate",0:"Graduate"},inplace=True)
df["Self_Employed"].replace({1:"Yes",0:"No"},inplace=True)
df["Property_Area"].replace({0:"Rural",1:"Semiurban",2:"Urban"},inplace=True)


# In[46]:


df.to_csv("F:\MACHINE LEARNING\Project/Final_output.csv")

