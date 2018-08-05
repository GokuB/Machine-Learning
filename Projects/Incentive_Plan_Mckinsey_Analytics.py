#Analytical Vidya
#McKinsey Analytics Online Hackathon
#21st July 2018
#Experiment by Gokul Balaji 
#GitHub: https://github.com/Gokul-Balaji
#LinkedIn: https://linkedin.com/in/gokulbalaji

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Policy=pd.read_csv('Train.csv')
Features=Policy.iloc[:, 0:11].values
Headers=np.array(Policy.columns)


#Data Exploration
Policy_ID=Policy.iloc[:,0:1].values
Paid=Policy.iloc[:,1:2].values
Age=Policy.iloc[:, 2:3].values
Income=Policy.iloc[:, 3:4].values
Count3to6=Policy.iloc[:, 4:5].values
Count6to12=Policy.iloc[:, 5:6].values
Underwriting_Score=Policy.iloc[:, 6:7].values
Premiums_Paid=Policy.iloc[:, 7:8].values
Sourcing=Policy.iloc[:, 8:9].values
Residence=Policy.iloc[:, 9:10].values
Premium=Policy.iloc[:, 10:11 ].values
Renewal_Status=Policy.iloc[:, 11:12].values









