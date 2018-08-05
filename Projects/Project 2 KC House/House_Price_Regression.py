# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:57:53 2018

@author: ItsMeK!
"""

#House Price Prediction 

#Importing Libraries for Data Manipulation and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

#Loading Data 
House=pd.read_csv("KC_Data.csv")

#Selecting Features
Features_List=["bedrooms","	bathrooms","sqft_living","sqft_lot	","floors","condition","grade","sqft_above","sqft_basement","yr_built","zipcode","sqft_living15","sqft_lot15"]
Predict_List=["price"]

#Creating Independent and Dependent Variables
Features=House.loc[:, Features_List].values
Price=House.loc[:, Predict_List].values

#Adding Missing Values to the Data

from sklearn.preprocessing import Imputer
Impute_Features=Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
Impute_Price=Imputer(missing_values="NaN", strategy="mean", axis=0)
Impute_Features.fit(Features)
Impute_Price.fit(Price)
Features=Impute_Features.transform(Features) 
Price=Impute_Price.transform(Price)

#Backward Elimination Process to eliminate uncorelated features
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Price, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    print (regressor_OLS.summary())
    return x
 
SL = 0.05
Features_Modeled = backwardElimination(Features, SL)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#Scale=StandardScaler()
#Features_Scaled=Scale.fit_transform(Features_Modeled)
#Predict_Scaled=Scale.fit_transform(Price)


#Splitting Training and Test Data
from sklearn.model_selection import train_test_split
Features_Train, Features_Test, Price_Train, Price_Test=train_test_split(Features_Modeled, Price,test_size=0.3, random_state=0)


#Creating Regression Model
from sklearn.linear_model import LinearRegression
Predictor=LinearRegression()
Predictor.fit(Features_Train, Price_Train)
Price_Predicted=Predictor.predict(Features_Test)


from sklearn.metrics import mean_squared_error, r2_score

print ("Coefficint: "+str(Predictor.coef_))
print ("Rank: "+str(Predictor.rank_))
#print ("Score: "+str (Predictor.score(Price_Test, Price_Predicted)))
print ("R Score of the Model: " + str(mean_squared_error(Price_Test, Predictor.predict(Features_Test))))
print ("R^2 Score of the Model: " + str(r2_score(Price_Test, Predictor.predict(Features_Test))))

OLS_Predictor=sm.OLS(Price_Train,Features_Train).fit()
print (OLS_Predictor.summary())









