#Data Exploration with Pandas
#https://www.analyticsvidhya.com/wp-content/uploads/2016/08/Data-Exploration-in-Python.pdf

#Loading Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading Dataset
df=pd.read_csv('IMDB-Movie-Data.csv')

#Data Exploration
#Preview of Dataframe
print (df.shape)
print (df.head(5)) #Returns top 5 rows
print (df.tail(5)) #Returns bottom 5 rows
Columns=df.columns #Returns columns of Dataframe
print (Columns)
print (df.info())
print (df.isnull().sum()) #Returns Number of Null Records/Columns
print (df.count()) #Returns Number of Not Null Records in Dataframe
print (df.dtypes) #Returns data type of Columns
print (df.describe()) #Statistical Analysis of Data
print (df.index) #Returns start and stop of record indices
print (df.values) #Returs values of Dataframe
print (df.T) #Transpose Data 

#Sorting Data 
print (df.sort_index(axis=0, ascending=False)) #Sort by Rows
print (df.sort_index(axis=1, ascending=False)) #Sort by Columns
print (df.sort_index(by='Metascore', ascending=False)) #Sort Column specifically 
print (df.sort_index(by=['Metascore'], ascending=False).sort_index(by=['Votes'], ascending=True)) #Sorting by 1+ columns
print (df.sort_index(by='Metascore', ascending=False).head(5)) #Sort by Column and return top 5
print (df.sort_index(by='Metascore', ascending=False).tail(5)) #Sort by Column and return bottom 5

#Selecting Data
print (df['Title']) #Individual Columns selection
print (df.iloc[:, 1]) #Selecting by position with [:-> rows, 1-> column position]
print (df.iloc[:, [1, 2, 3, 4]]) #Selects all rows of column 1 to 4
print (df.iloc[:, 2].unique()) #Unique values of columns
print (df.iloc[:, [2,-2]].sort_index())
print (df.sort_index(by='Revenue', ascending=False).iloc[:, [2, -2]].head(5)) #Sort and Selection
print (df.iloc[:, [1, -3, -2, -1]].sort_index(by='Votes', ascending=False).head(5)) #Select and Sort

#Filtering Data
print (df[df.Metascore>90].sort_index(by='Metascore',ascending=False)) #Filter and Sort 
print (df[df.Votes>1000000].iloc[:, [1, -3, -2, -1]].sort_index(by='Title')) #Filter Select and Sort
print (df[df.Revenue>500].sort_index(by='Votes', ascending=False)) 

#Grouping Data
print (df.iloc[:, [6,  -3, -2, -1]][df.Year>2015].groupby('Year').sum())
print (df.loc[:, ['Year', 'Genre']].groupby(['Year']).count())
print ((df[(df.Metascore>90) & (df.Votes>10000)]).loc[:, ['Year', 'Genre', 'Votes', 'Revenue', 'Metascore']].groupby(['Year', 'Genre']).sum())

#Statistics
print (df.describe()) #Statistical Analysis on Dataset
print (df.cov()) #Covarince between two columns
print (df.corr()) #Correlation between columns

#Handling Missing Values
print (df.isnull().sum()) 
dfc=df.copy()
dfc2=df.copy()
print (dfc.isnull().sum())
dfc=dfc.dropna() #Drop all the records that has Null Values
print (dfc.isnull().sum()) 
dfc2=df.fillna(value=5) #Fill all Null Values with certain value
print (dfc2.describe())
print (df['Year'].value_counts())
print (df['Votes'].mean())
print (df['Runtime'].median())
print (df['Year'].median())

#Datatype Conversion 

print (df.dtypes)
dfc['Runtime']=dfc['Runtime'].astype(np.float)
print (dfc.dtypes)


#Functions
print (df.apply(np.max))
print (df[df['Director'].apply(lambda item: item=='Steven Spielberg')]) #Using Lambda and Filter
print (df[df['Genre'].apply(lambda item: item=='Comedy')])
d={'Comedy':'Romantic Comedy', 'Action':'Adventure'}
print (df[(df.Genre=='Action') | (df.Genre=='Comedy')].count())
print (df.replace({'Genre':d}))

#Summary in Pandas
print (pd.crosstab(df['Genre'], df['Year']))
print (df.pivot_table(['Revenue'],['Year'] , aggfunc='sum')) 

#Visual Plots Histogram, Count Plot, Scatter, Box Plot

Fig=plt.figure()
plt.hist(df['Year'], bins=10)
plt.show()

plt.scatter(df['Genre'], df['Revenue'])
plt.show()

house=pd.read_csv('kc_house_data.csv')
print (house.shape)
print (house.columns)
print (house.isnull().sum())
house=house.dropna()
print (house.shape)
print (house.isnull().sum())


plt.scatter(house['floors'].iloc[:50], house['price'].iloc[:50])
plt.show()

plt.hist(house['floors'], bins=5)
plt.show()

import seaborn as sns
plt.boxplot(house['price'])
plt.show()

sns.boxplot(df['Revenue'])
sns.despine()






