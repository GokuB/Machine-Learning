#Logistic Regression Model to determine Insurance Renewal


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading Training Dataset into the Model
Policy=pd.read_csv('Train.csv')
Feature_List=Policy.columns.values
Features=Policy.iloc[:, 0:len(Feature_List)-1].values
Renewal_Status=Policy.loc[:, ["renewal"]].values
Policy_Copy=Policy

#Identifying Missing Values
Policy.shape
Policy.isnull().sum() #Count of NaN values per column
Policy.dtypes #Datatype of the column

#Loading Testing Dataset into the Model
Testing=pd.read_csv('Test.csv')
Testing.shape
Testing.isnull().sum()
Testing.dtypes

Test_Features=Testing.iloc[:, 0:len(Feature_List)-1].values


#Adding Missing value to the Training Dataset 
from sklearn.preprocessing import Imputer
Count_Months=Imputer(missing_values="NaN", strategy="median", axis=0)
Count_Months=Count_Months.fit(Features[:, [4, 5, 6]])
Features[:, [4, 5, 6]]=Count_Months.transform(Features[:, [4, 5, 6]])

Application_Score=Imputer(missing_values="NaN", strategy="mean", axis=0)
Application_Score=Application_Score.fit(Features[:, [7]])
Features[:, [7]]=Application_Score.transform(Features[:, [7]])

#Adding Missing value to the Testing Dataset 
from sklearn.preprocessing import Imputer
Count_Months=Imputer(missing_values="NaN", strategy="median", axis=0)
Count_Months=Count_Months.fit(Test_Features[:, [4, 5, 6]])
Test_Features[:, [4, 5, 6]]=Count_Months.transform(Test_Features[:, [4, 5, 6]])

Application_Score=Imputer(missing_values="NaN", strategy="mean", axis=0)
Application_Score=Application_Score.fit(Test_Features[:, [7]])
Test_Features[:, [7]]=Application_Score.transform(Test_Features[:, [7]])

#Validating the Missing Value
pd.DataFrame(Features).isnull().sum()
pd.DataFrame(Test_Features).isnull().sum()

#Encoding Categorical Variables
Sourcing_Category=["sourcing_channel"]
Residence_Category=["residence_area_type"]

from sklearn.preprocessing import LabelEncoder
Sourcing_Encode=LabelEncoder()
Sourcing_Encode=Sourcing_Encode.fit(Features[:, 9])
Features[:,9]=Sourcing_Encode.transform(Features[:, 9])

Residence_Encode=LabelEncoder()
Residence_Encode=Residence_Encode.fit(Features[:, 10])
Features[:,10]=Residence_Encode.transform(Features[:, 10])

from sklearn.preprocessing import LabelEncoder
Sourcing_Encode=LabelEncoder()
Sourcing_Encode=Sourcing_Encode.fit(Test_Features[:, 9])
Test_Features[:,9]=Sourcing_Encode.transform(Test_Features[:, 9])

Residence_Encode=LabelEncoder()
Residence_Encode=Residence_Encode.fit(Test_Features[:, 10])
Test_Features[:,10]=Residence_Encode.transform(Test_Features[:, 10])


#Feature Scaling
from sklearn.preprocessing import StandardScaler
Scale_Train=StandardScaler()
Train_Features_Scaled=Scale_Train.fit_transform(Features)
Scale_Test=StandardScaler()
Test_Features_Scaled=Scale_Test.fit_transform(Test_Features)


#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
Classifier=LogisticRegression()
Classifier=Classifier.fit(Train_Features_Scaled, Renewal_Status)
Renewal_Predicted=Classifier.predict(Test_Features_Scaled)

Probability_Test=Classifier.predict_proba(Test_Features_Scaled)
Probability_Train=Classifier.predict_proba(Train_Features_Scaled)

Policy_Probabilty=[[0 for j in range(3)] for i in range(len(Features))]
for i in range(len(Features)):
    Policy_Probabilty[i][0]=Features[i, 0]
    Policy_Probabilty[i][1]=Probability_Train[i, 1]
    Policy_Probabilty[i][2]=Policy.iloc[i, 11]

Policy_Probabilty=np.array(Policy_Probabilty)

#Save as Output
Policy_Df=pd.DataFrame(Policy_Probabilty)
Writer=pd.ExcelWriter("Policy_Probabiltity_Premium.xlsx")
Policy_Df.to_excel(Writer, "Policy Prediction")
Writer.save()
    


    








