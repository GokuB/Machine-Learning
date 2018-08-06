#Open Machine Learning Course
#Topic 2. Visual Data Analysis with Python

#Initializing the Environment 
import pandas as pd
import numpy as np
pd.options.display.max_columns=12

#Disable Warnings in Anaconda
import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format='svg'

#Increase the default plot size
from pylab import rcParams
rcParams['figure.figsize']=5, 4

#Loading Data
df=pd.read_csv('telecom_churn.csv')
dfc=pd.read_csv('telecom_churn.csv')
df.shape
df.head()
df.describe()
df.info()
df.columns
#Univariate visualization
#Histogram Plot
features=['Total day calls', 'Total eve calls', 'Total night calls','Total intl calls']
df[features].hist(figsize=(12,4), bins=10)

#Density Plot
df[features].plot(kind='density', subplots=True, layout=(1, 4), sharex=False, figsize=(12, 4))

sns.distplot(df['Total day calls'])
sns.distplot(df['Total eve calls'])
sns.distplot(df['Total night calls'])
sns.distplot(df['Total intl calls'])

#Boxplot
_, ax=plt.subplots(figsize=(3, 4))
_, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.boxplot(df['Total intl calls'], ax=axes[0])
sns.violinplot(df['Total intl calls'], ax=axes[1])
df['Total intl calls'].describe()

# Categorical and binary features
df['Churn'].value_counts()
df['International plan'].value_counts()
df['Voice mail plan'].value_counts()

#Bar Plot
_, axes=plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
sns.countplot(x='Customer service calls', data=df,ax=axes[0])
sns.countplot(x='Churn', data=df, ax=axes[1])

'''
Histograms are best suited for looking at the distribution of numerical variables while bar plots are used for categorical features.
The values on the X-axis in the histogram are numerical; a bar plot can have any type of values on the X-axis: numbers, strings, booleans.
The histogram’s X-axis is a Cartesian coordinate axis along which values cannot be changed; the ordering of the bars is not predefined. Still, it is useful to note that the bars are often sorted by height, that is, the frequency of the values. Also, when we consider ordinal variables (like Customer service calls in our data), the bars are usually ordered by variable value.
'''

#Multivariate visualization

#Quantitative–Quantitative
#Correlation matrix
df.corr()
#Heatmap to visualize the correlation between the variables


#Removing Non-Numerical Variables
non_num=set(['State','International plan','Voice mail plan','Churn'])
Non=list(set(df.columns)-non_num)
corr_mat=df[Non].corr()
sns.heatmap(corr_mat)

Dep=set(['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge'])
Numerical=list(set(Non)-Dep)
corr_mat2=df[Numerical].corr()
sns.heatmap(corr_mat2)

#Scatter Plot
plt.scatter(df['Total day minutes'], df['Total night minutes'])


#Joint Plot
sns.jointplot(x=df['Total day minutes'],y=df['Total night minutes'], data=df, kind='scatter')
sns.jointplot(x=df['Total day minutes'],y=df['Total night minutes'], data=df, kind='reg')
sns.jointplot(x=df['Total day minutes'],y=df['Total night minutes'], data=df, kind='resid')
sns.jointplot(x=df['Total day minutes'],y=df['Total night minutes'], data=df, kind='kde', color='r')
sns.jointplot(x=df['Total day minutes'],y=df['Total night minutes'], data=df, kind='hex')

#Scatterplot matrix
# pairplot may become very slow with the SVG format
%config InlineBackend.figure_format = 'png'
sns.pairplot(df[Numerical])

#Quantitative–Categorical

sns.lmplot('Total day minutes', 'Total night minutes', data=df, hue='State', fit_reg=False)
sns.lmplot('Total day minutes', 'Total night minutes', data=df, hue='Churn', fit_reg=False)

fig, axes=plt.subplots(nrows=3, ncols=4, figsize=(20, 10))
for idx, feat in enumerate(Numerical): 
    ax = axes[int(idx / 4), idx % 4] 
    sns.boxplot(x='Churn', y=feat, data=df, ax=ax) 
    ax.set_xlabel('') 
    ax.set_ylabel(feat) 
fig.tight_layout()


sns.factorplot(x='Churn', y='Total day minutes',
               col='Customer service calls',
               data=df[df['Customer service calls'] < 8], 
               kind="box", col_wrap=4, size=3, aspect=.8)
    
sns.countplot(x='Customer service calls', hue='Churn', data=df)

_, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4)) 
sns.countplot(x='International plan', hue='Churn', data=df, ax=axes[0])
sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=axes[1])

#Contingency table

pd.crosstab(df['State'], df['Churn']).T
df.groupby(['State'])['Churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T

#Dimensionality reduction
#t-distributed Stohastic Neighbor Embedding

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

df.info()
df.head()
Map={'Yes':1, 'No':0}
df['International plan']=df['International plan'].map(Map)
df['Voice mail plan']=df['Voice mail plan'].map(Map)
df=df.drop(['State', 'Churn'], axis=1)

#Normalization (Subtract Mean from variable Divide by Standard Deviation)

Scale=StandardScaler()
dfs=Scale.fit_transform(df)

#Building tsne representation

%%time tsne=TSNE(random_state=17)
ts=TSNE()
tsne_rep=ts.fit_transform(dfs)


plt.scatter(tsne_rep[:, 0], tsne_rep[:, 1])
plt.scatter(tsne_rep[:, 0], tsne_rep[:, 1], color=dfc['Churn'].map({False:'Red', True:'Yellow'}))

_, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
for i, name in enumerate(['International plan', 'Voice mail plan']):
    axes[i].scatter(tsne_rep[:, 0], tsne_rep[:, 1],
                    c=dfc[name].map({'Yes': 'green', 'No': 'red'}))
    axes[i].set_title(name)

