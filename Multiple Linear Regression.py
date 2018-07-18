# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer #For Missing Values in the DataSet 
from sklearn.preprocessing import OneHotEncoder #To create Dummy Variables
from sklearn.preprocessing import LabelEncoder #To Encode Categorical Variables
from sklearn.preprocessing import StandardScaler #To Feature Scale
from sklearn.model_selection import train_test_split #To split Test and Train Data
from sklearn.linear_model import LinearRegression #Regression Model 
import statsmodels.formula.api as sm #OLS implementation
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score #To Calculate metrics

#Importing Dataset
from sklearn.datasets import load_boston #Using Boston Dataset from SciKit Learn
Boston=load_boston()

#Exploring Data
print (Boston.keys()) #dict_keys(['data', 'target', 'feature_names', 'DESCR'])
print (Boston.data.shape) #(506, 13)
print (Boston.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'  'B' 'LSTAT']
print (Boston.DESCR)

#Features List
'''
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's 
'''
#Creating Features and Prediction Attributes from the Dataset
Features=pd.DataFrame(Boston.data).values
Prediction=pd.DataFrame(Boston.target).values
Features_List=pd.DataFrame(Boston.feature_names)

#Statistical Analysis on the Dataset
Description=pd.DataFrame(Boston.data).describe()

#Feature Scaling
Scaler1=StandardScaler()
Scaler2=StandardScaler()

Scaler1.fit(Features)
Features_Scaled=Scaler1.transform(Features)

Scaler2.fit(Prediction)
Prediction_Scaled=Scaler2.transform(Prediction)


#Splting Dataset for Training and Testing
Features_Train, Features_Test, Prediction_Train, Prediction_Test= train_test_split(Features_Scaled, Prediction_Scaled, test_size=0.2,random_state=0 )

#Training Data
print (Features_Train.shape) #(404, 13)

#Testing Data
print (Features_Test.shape) #(102, 13)

#Regression Model Creation 
Regressor=LinearRegression()

#Fitting Training Dataset to the Model
Regressor.fit(Features_Train, Prediction_Train)

#Predicting output with Test Dataset 
Predicted=Regressor.predict(Features_Test)

print (Prediction_Test.shape) #(102, 1)
print (Predicted.shape) #(102, 1)

#Accuracy of the Regression Model

print ("Accuracy of the Multiple Linear Regression Model")
print ("Coef"+str(Regressor.coef_))
print ("Rank:"+str(Regressor.rank_))
#print (Regressor.score(Prediction_Test, Predicted))

#Accuracy of the Model

print("Mean Squared Error:"+str(mean_squared_error(Prediction_Test, Predicted)))
print ("R^2 Score:"+str(r2_score(Prediction_Test, Predicted)))

#Backward Elimination Process to Eliminate uncoreleated features


def backwardElimination(x,y,sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print (regressor_OLS.summary())
    return x

P_Value=0.05
Features_Modeled=backwardElimination(Features,Prediction,P_Value)

Features_ModelTrain, Features_ModelTest, Prediction_ModelTrain, Prediction_ModelTest= train_test_split(Features_Modeled, Prediction_Scaled, test_size=0.2,random_state=0 )

Modeled_Regressor=LinearRegression()
Modeled_Regressor.fit(Features_ModelTrain, Prediction_ModelTrain)
Modeled_Predicted=Modeled_Regressor.predict(Features_ModelTest)

print ("Accuracy of the Multiple Linear Regression Model (Backward Elimination)")
print ("Coef:"+str(Modeled_Regressor.coef_))
print ("Rank:"+str(Modeled_Regressor.rank_))
print ("R^2 Score"+str(r2_score(Prediction_ModelTest, Modeled_Predicted)))
print ("Mean Squared Error:"+str(mean_squared_error(Prediction_ModelTest, Modeled_Predicted)))


#Prediction Value Plot (Test Data vs Predicted Data without backward elimination)
X_grid = np.arange(0, 102, 1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_grid,Prediction_Test, color="green", label="Test Data")
plt.scatter(X_grid,Predicted, color="red", label='Predicton Scaled')
plt.ylabel("Predicted Values")
plt.xlabel("Scale")
plt.legend(loc='upper left')
plt.show()

#Prediction Value Plot (Test Data vs Predicted Data with backward elimination)
X_grid = np.arange(0, 102, 1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_grid,Prediction_Test, color="green", label="Test Data")
plt.scatter(X_grid,Modeled_Predicted, color="blue", label='Prediction Modeled (OLS)')
plt.ylabel("Predicted Values")
plt.xlabel("Scale")
plt.legend(loc='upper left')
plt.show()

