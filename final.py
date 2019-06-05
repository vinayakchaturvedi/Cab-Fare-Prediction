# -*- coding: utf-8 -*-
"""
Created on Sun May 19 23:45:55 2019

@author: vinayak
"""
#Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from fancyimpute import KNN
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency

################################## User defined Functions ################################################
def divideDateTime(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], errors='coerce', format='%Y-%m-%d %H:%M:%S UTC')
    #dividepickup_datetime into parts = date + month + year + day
    data['day']=data['pickup_datetime'].apply(lambda x:x.day)
    data['month']=data['pickup_datetime'].apply(lambda x:x.month)
    data['year']=data['pickup_datetime'].apply(lambda x:x.year)
    data['dayOfWeek']=data['pickup_datetime'].apply(lambda x:x.weekday())
    data['hour']=data['pickup_datetime'].apply(lambda x:x.hour)
    data = data.drop(["pickup_datetime"], axis=1) 
    return data
        
    
def convertIntoProperDataTypes(data):
    if(data.shape[1]==11):
        cnumber_factor = [6,7,8,9,10]
    else:
        cnumber_factor = [5,6,7,8,9]
    for i in cnumber_factor:
        data.iloc[:,i] = data.iloc[:,i].astype('object')
    return data
        
def getCnamesFactor(data):
    lis = []
    for i in range(0, data.shape[1]):
        if(data.iloc[:,i].dtypes == 'object'):
            lis.append(data.columns[i])
    return lis

def getCnamesNumeric(data):
    lis = []
    for i in range(0, data.shape[1]):
        if(data.iloc[:,i].dtypes != 'object'):
            lis.append(data.columns[i])
    return lis
    
def getOptimalImputeMethod(data):
    bestFit = {'Actual':-73.99578100000001}
    data['pickup_longitude'].loc[70] = np.nan
    data_knn = pd.DataFrame.copy(data)
    data_mean= pd.DataFrame.copy(data)
    data_median= pd.DataFrame.copy(data)
    print(data_median['pickup_longitude'].loc[70])
    #Impute with mean
    data_mean['pickup_longitude'] = data_mean['pickup_longitude'].fillna(data_mean['pickup_longitude'].mean())
    print(data_median['pickup_longitude'].loc[70])
    #Impute with median
    data_median['pickup_longitude'] = data_median['pickup_longitude'].fillna(data_median['pickup_longitude'].median())
    #Impute with KNN
    data_knn = pd.DataFrame(KNN(k = 3) .fit_transform(data_knn), columns = data_knn.columns)

    bestFit['Using KNN'] = data_knn['pickup_longitude'].loc[70]
    bestFit['Using Mean'] = data_mean['pickup_longitude'].loc[70]
    bestFit['Using Median'] = data_median['pickup_longitude'].loc[70]
    return bestFit;

def imputeMissingValues(data, cnames_numeric, cnames_factor):
    for i in cnames_numeric:
        data[i] = data[i].replace(0, np.nan)
      
    #KNN imputation
    #Assigning levels to the categories
    lis = []
    for i in range(0, data.shape[1]):
        if(data.iloc[:,i].dtypes == 'object'):
            data.iloc[:,i] = pd.Categorical(data.iloc[:,i])
            data.iloc[:,i] = data.iloc[:,i].cat.codes 
            data.iloc[:,i] = data.iloc[:,i].astype('object')
            lis.append(data.columns[i])
    #replace -1 with NA to impute
    for i in range(0, data.shape[1]):
        data.iloc[:,i] = data.iloc[:,i].replace(-1, np.nan) 
    #Apply KNN imputation algorithm
    data = pd.DataFrame(KNN(k = 3) .fit_transform(data), columns = data.columns)
    #Convert into proper datatypes
    for i in lis:
        data.loc[:,i] = data.loc[:,i].round()
        data.loc[:,i] = data.loc[:,i].astype('object')
    data.passenger_count = data.passenger_count.round()
    #missing_val = pd.DataFrame(data.isnull().sum())
    return data

def checkOutlier(data, cnames_numeric):
    number_of_columns=len(cnames_numeric)
    number_of_rows = len(cnames_numeric)-1/number_of_columns
    plt.figure(figsize=(number_of_columns,5*number_of_rows))
    for i in range(0,len(cnames_numeric)):
        plt.subplot(number_of_rows + 1,number_of_columns,i+1)
        sns.set_style('whitegrid')
        sns.boxplot(data[cnames_numeric[i]],color='green',orient='v')
        plt.tight_layout()

def removeOutlier(data, cnames_numeric, cnames_factor):
    for i in cnames_numeric:
        q75, q25 = np.percentile(data.loc[:,i], [75 ,25])
        #Calculate IQR
        iqr = q75 - q25
        #Calculate inner and outer fence
        minimum = q25 - (iqr*1.5)
        maximum = q75 + (iqr*1.5)
        #Replace with NA
        data.loc[data.loc[:,i] < minimum,i] = np.nan
        data.loc[data.loc[:,i] > maximum,i] = np.nan


    #Apply KNN imputation algorithm
    data = pd.DataFrame(KNN(k = 3) .fit_transform(data), columns = data.columns)
    #Convert into proper datatypes
    for i in cnames_factor:
        data.loc[:,i] = data.loc[:,i].round()
        data.loc[:,i] = data.loc[:,i].astype('object')
    data.passenger_count = data.passenger_count.round()
    #missing_val = pd.DataFrame(data.isnull().sum())
    return data
    
def featureScaling(data, cnames_numeric):
    for i in cnames_numeric:
        print(i)
        data[i] = (data[i] - data[i].mean())/data[i].std()
    return data

################################## User defined Functions End ##########################################

#Set directory
os.chdir("C:/Users/vinayak\Desktop/Car Fare prediction")

################################## Load dataset ########################################################
train = pd.read_csv("train_cab.csv")
test = pd.read_csv("test.csv")

################################## Data Cleaning and preparation ########################################
train.info()
train['fare_amount'] = pd.to_numeric(train['fare_amount'], errors='coerce')
train.info()
test.info()

train = divideDateTime(train)
test = divideDateTime(test) 
    
train = convertIntoProperDataTypes(train)
test = convertIntoProperDataTypes(test)
    
cnames_numeric = getCnamesNumeric(train)
cnames_factor = getCnamesFactor(train)

train.info()
test.info() 
    
################################## Missing Value Analysis ################################################
bestImpute = getOptimalImputeMethod(train)

train = imputeMissingValues(train, cnames_numeric, cnames_numeric)

missing_val = pd.DataFrame(train.isnull().sum())
train = convertIntoProperDataTypes(train)
################################## Analyse Data Distribution and Graphs ###################################
train.describe()

#Analyze Distribution
number_of_columns=6
number_of_rows = len(cnames_numeric)-1/number_of_columns
plt.figure(figsize=(7*number_of_columns,5*number_of_rows))
for i in range(0,len(cnames_numeric)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.kdeplot(train[cnames_numeric[i]]).set_title("Distribution of "+cnames_numeric[i])

plt.figure(figsize=(7*number_of_columns,5*number_of_rows))
for i in range(0,len(cnames_numeric)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.kdeplot(np.log(train[cnames_numeric[i]].values)).set_title("Distribution of "+cnames_numeric[i] + "Using log scale")

#Analyze Distribution of Latitude-Longitude points
city_lat_border = (40.50, 41.00)
city_long_border = (-74.15, -73.60)
train.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', color='green')
plt.title("Dropoff locations")
plt.ylim(city_lat_border)
plt.xlim(city_long_border)

train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',color='blue')
plt.title("Pickups")
plt.ylim(city_lat_border)
plt.xlim(city_long_border)

#Relation Between Categorical variable and target
plt.figure(figsize=(8,6))
sns.barplot(data=train, x="year", y="fare_amount")

plt.figure(figsize=(12,6))
sns.barplot(data=train, x="month", y="fare_amount")

plt.figure(figsize=(15,6))
sns.barplot(data=train, x="day", y="fare_amount")

plt.figure(figsize=(8,6))
sns.barplot(data=train, x="dayOfWeek", y="fare_amount")

plt.figure(figsize=(8,6))
sns.barplot(data=train, x="hour", y="fare_amount")

################################## Outlier Analysis ######################################################
checkOutlier(train, cnames_numeric)

train = removeOutlier(train, cnames_numeric, cnames_factor)
train = convertIntoProperDataTypes(train)

################ Analyse Data Distribution and Graphs (After Outlier Analysis) ###########################
train.describe()

#Analyze Distribution
number_of_columns=6
number_of_rows = len(cnames_numeric)-1/number_of_columns
plt.figure(figsize=(7*number_of_columns,5*number_of_rows))
for i in range(0,len(cnames_numeric)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.kdeplot(train[cnames_numeric[i]]).set_title("Distribution of "+cnames_numeric[i])

#Analyze Distribution
plt.figure(figsize=(5*number_of_columns,8*number_of_rows))
for i in range(0,len(cnames_numeric)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.distplot(train[cnames_numeric[i]],kde=True) 

#Analyze Distribution of Latitude-Longitude points
city_lat_border = (40.50, 41.00)
city_long_border = (-74.15, -73.60)
train.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', color='green')
plt.title("Dropoff locations")
plt.ylim(city_lat_border)
plt.xlim(city_long_border)

train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',color='blue')
plt.title("Pickups")
plt.ylim(city_lat_border)
plt.xlim(city_long_border)

#Relation Between Categorical variable and target
plt.figure(figsize=(8,6))
sns.barplot(data=train, x="year", y="fare_amount")

plt.figure(figsize=(12,6))
sns.barplot(data=train, x="month", y="fare_amount")

plt.figure(figsize=(15,6))
sns.barplot(data=train, x="day", y="fare_amount")

plt.figure(figsize=(8,6))
sns.barplot(data=train, x="dayOfWeek", y="fare_amount")

plt.figure(figsize=(10,6))
sns.barplot(data=train, x="hour", y="fare_amount")

#Relation Between Categorical variable and number of passengers
plt.figure(figsize=(8,6))
sns.barplot(data=train, x="year", y="passenger_count")

plt.figure(figsize=(12,6))
sns.barplot(data=train, x="month", y="passenger_count")

plt.figure(figsize=(15,6))
sns.barplot(data=train, x="day", y="passenger_count")

plt.figure(figsize=(8,6))
sns.barplot(data=train, x="dayOfWeek", y="passenger_count")

plt.figure(figsize=(10,6))
sns.barplot(data=train, x="hour", y="passenger_count")
################################## Feature Selection ################################################
df_corr = train.loc[:,cnames_numeric]
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(10, 10))
#Generate correlation matrix
corr = df_corr.corr()
#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap = 'viridis',
            square=True, ax=ax, annot=True)


for i in range(0,len(cnames_factor)):
    for j in range(i+1,len(cnames_factor)):
        print(cnames_factor[i], " VS ", cnames_factor[j])
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(train[cnames_factor[i]], train[cnames_factor[j]]))
        print(p)

#No variable is highly correlated to any other variable so don't remove any variable

################################## Feature Scaling ################################################
cnames_numeric.remove("fare_amount")
train = featureScaling(train, cnames_numeric)
test = featureScaling(test, getCnamesNumeric(test))

################################## Model Development ################################################
#Divide data into train and test
X = train.values[:, 1:]
Y = train.values[:,0]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)
   
#Linear Regression
regressor = LinearRegression()     
regressor.fit(X_train, y_train)                      #Fit the model on training data
y_pred = regressor.predict(X_test)                   #predicting the test set results  

lm_rmse=np.sqrt(mean_squared_error(y_pred, y_test))
print("Test RMSE for Linear Regression is ",lm_rmse)    #4.003082145405653
        
#Decision tree for regression
regressor = DecisionTreeRegressor(max_depth =6, random_state = 0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
dt_rmse=np.sqrt(mean_squared_error(y_pred, y_test))
print("Test RMSE for Decision tree is ",dt_rmse)    #3.5597152258765834

#Random Forest
regressor = RandomForestRegressor(n_estimators = 100, random_state = 883,n_jobs=-1)
regressor.fit(X_train,y_train)
y_pred= regressor.predict(X_test)
rf_rmse=np.sqrt(mean_squared_error(y_pred, y_test))
print("RMSE for Random Forest is ",rf_rmse)      #2.5970888685703235

#Random Forest :::: RMSE = 2.5970888685703235 --> best among all  
result = regressor.predict(test)
resultDataFrame = pd.concat([pd.DataFrame({"fare_amount":result}).reset_index(drop=True), test], axis=1)
#Note: RMSE may vary as training and test data may vary.
