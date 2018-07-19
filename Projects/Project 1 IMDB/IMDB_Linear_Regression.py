# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 21:20:26 2018

@author: ItsMeK!
"""

#IMDB- Meta Score Calculation


#Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Predict_Scaled, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print (regressor_OLS.summary())
    return x

#Loading Dataset 
IMDB=pd.read_csv('IMDB-Movie-Data.csv')
IMDB.describe()

#Selecting needed columns from Dataframe
Features_List=["Rank", "Year", "Runtime", "Rating", "Votes", "Revenue"]
Predict_List=["Metascore"]
Features=IMDB.loc[:, Features_List].values
Predict=IMDB.loc[:, Predict_List].values

#Acquiring Features of the dataset
Year=IMDB.loc[:, ["Year"]].values
Rank=IMDB.loc[:, ["Rank"]].values
Rating=IMDB.loc[:, ["Rating"]].values
Revenue=IMDB.loc[:, ["Revenue"]].values
Votes=IMDB.loc[:, ["Votes"]].values
Metascore=IMDB.loc[:, ["Metascore"]].values


#Visualizing the Data spread

#Figure=plt.figure()
#plt.subplot(231)
#plt.scatter(Rank, Rating, color='red')
#plt.subplot(232)
#plt.scatter(Rank, Revenue, color='blue')
#plt.subplot(233)
#plt.scatter(Rank, Metascore, color='green')
#plt.subplot(234)
#plt.scatter(Rank, Votes, color='black')
#plt.show()


#Adding Missing Values to the Columns
 
from sklearn.preprocessing import Imputer
imputer_features=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer_predict=Imputer(missing_values="NaN", strategy="mean", axis=0)

imputer_features=imputer_features.fit(Features)
Features=imputer_features.transform(Features)

imputer_predict=imputer_predict.fit(Predict)
Predict=imputer_predict.transform(Predict)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
Scale=StandardScaler()
Features_Scaled=Scale.fit_transform(Features)
Predict_Scaled=Scale.fit_transform(Predict)
 

#Applying Backward Elimination
import statsmodels.formula.api as sm

SL = 0.05
Features_Modeled = backwardElimination(Features, SL)

#Spliting the Dataset for Training and Testing
from sklearn.model_selection import train_test_split
Features_Train, Features_Test, Predict_Train, Predict_Test=train_test_split(Features_Scaled, Predict_Scaled, test_size=0.2, random_state=0)

#Creating Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
Metascore_Predictor=LinearRegression()
Metascore_Predictor.fit(Features_Train, Predict_Train)
Metascore_Predicted=Metascore_Predictor.predict(Features_Test)

#Regression Analysis

from sklearn.metrics import mean_squared_error, r2_score

print ("R Score of the Model: " + str(mean_squared_error(Predict_Test, Metascore_Predicted)))
print ("R^2 Score of the Model: " + str(r2_score(Predict_Test, Metascore_Predicted)))
print (Metascore_Predictor.coef_)
print (Metascore_Predictor.intercept_)
print (Metascore_Predictor.rank_)


#Creating Polynomial Regression Model

Metadata_Poly=LinearRegression()

from sklearn.preprocessing import PolynomialFeatures

Poly=PolynomialFeatures(2)
Poly.fit(Features_Scaled)
Features_Poly=Poly.fit_transform(Features_Scaled)

Features_PolyTrain, Features_PolyTest, Predict_PolyTrain, Predict_PolyTest=train_test_split(Features_Poly, Predict_Scaled, test_size=0.2, random_state=0)

Metadata_Poly.fit(Features_PolyTrain, Predict_PolyTrain)
Predict_Poly=Metadata_Poly.predict(Features_PolyTest)

print ("R Score of the Model: " + str(mean_squared_error(Predict_PolyTest, Predict_Poly)))
print ("R^2 Score of the Model: " + str(r2_score(Predict_PolyTest, Predict_Poly)))
print (Metascore_Predictor.coef_)
print (Metascore_Predictor.intercept_)
print (Metascore_Predictor.rank_)















