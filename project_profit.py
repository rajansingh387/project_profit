

import pandas as pd
import numpy as np

data1= pd.read_json('Order_breakdown.json')

data1.head()

data1.columns

data1.shape

value=data1['Order ID'].value_counts()

value[value==2].value_counts()

data2= pd.read_csv('Order.tsv',sep='\t')

data2.shape

data2['Order ID'].value_counts()

data2.isnull().sum()

data2.head()

df =pd.merge(data1,data2,on='Order ID')

df.shape

df.isnull().sum()

print('total duplicated before',df.duplicated().sum())
df.drop_duplicates(inplace= True)
print('total duplicated after',df.duplicated().sum())

df.shape

df.sample()

import matplotlib.pyplot as plt
import seaborn as sns

df['Segment'].value_counts()

# creating a for loop to check unique values of all the columns if values of the unique value is less than 10 we will print value count of
#unique values too else we will just show how many unique values are there.
print(df.columns.value_counts().sum())
for i in df.columns:
    # Check if the number of unique values in the column is less than 10

    if df[i].nunique() < 10:
        # Print column information, including the column name, data type, number of unique values, and value counts
        print(f'The column "{i}" is __{df[i].dtype}__ \nhas __{df[i].nunique()}__ unique values: \n{df[i].value_counts()}')
        # Print a separator line
        print(40*'_')
    else:
        # Print column information, including the column name, data type, and number of unique values
        print(f'The column "{i}" is __{df[i].dtype}__ \nhas __{df[i].nunique()}__ unique values')
        # Print a separator line
        print(40*'_')

dfc= df.select_dtypes(include='object')

dfc.columns

dfc=dfc[[ 'Category', 'Sub-Category',
       'Country', 'Region', 'Segment',
       'Ship Mode']]

x = 0
fig = plt.figure(figsize=(20, 25))  # Create a figure with the specified size
plt.subplots_adjust(wspace=0.4)  # Adjust the spacing between subplots

# Iterate over each column in the DataFrame df_c
for i in dfc.columns:
    ax = plt.subplot(421 + x)  # Create a subplot at position 321+x in a 3x2 grid
    ax = sns.countplot(data=dfc, y=i)  # Create a countplot using data from df_c with y-axis representing the column values
    plt.grid(axis='x')  # Add gridlines along the x-axis
    ax.set_title(f'Distribution of {i}')  # Set the title of the subplot
    x += 1  # Increment x to move to the next subplot position



df.head()

df['Margin']= df['Profit']/df['Sales'] *100

# Assuming you have imported pandas and have a DataFrame named df
plt.figure(figsize=(16, 12))  # Set the size of the entire figure

# First subplot (left side)
plt.subplot(2, 1,1)
sns.barplot(data=df, x='Country', y='Margin',hue='Category')
plt.title('Margin')
plt.legend()

# Second subplot (right side)
plt.subplot(2,1, 2)
sns.barplot(data=df, x='Country', y='Sales',hue='Category')
plt.title('Sales')
plt.legend()
  # Adjust spacing between subplots to prevent overlapping
plt.show()

plt.figure(figsize=(25, 14))
ax= plt.subplot(2,1,1)
sns.barplot(x=df['Sub-Category'], y=df['Margin'],hue= df['Region'])

ax= plt.subplot(2,1,2)
sns.barplot(x=df['Sub-Category'], y=df['Sales'], hue=df['Region'])

plt.figure(figsize=(25, 14))
ax=plt.subplot(2,1,1)
sns.barplot(x=df['Sub-Category'], y=df['Sales'],errorbar=None)

ax=plt.subplot(2,1,2)
sns.barplot(x=df['Sub-Category'], y=df['Actual Discount'],errorbar=None)
plt.show()
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m-%d-%y')
df['Month'] = df['Order Date'].dt.strftime('%B')
df['Year'] = pd.to_datetime(df['Order Date']).dt.year

plt.figure(figsize=(25, 14))
ax=plt.subplot(2,1,1)
sns.barplot(x=df['Month'], y=df['Sales'],hue=df['Year'],errorbar=None)

ax=plt.subplot(2,1,2)
sns.barplot(x=df['Month'], y=df['Margin'],errorbar=None,hue=df['Year'])
plt.show()

plt.figure(figsize=(25, 14))

# Subplot 1: Sales with hue based on both Year and Category
ax = plt.subplot(2, 1, 1)
sns.barplot(x='Month', y='Sales',hue='Category' ,data=df, errorbar=None)

# Subplot 2: Margin with hue based on both Year and Category
ax = plt.subplot(2, 1, 2)
sns.barplot(x='Month', y='Margin', hue= 'Category',  data=df, errorbar=None)

plt.show()

plt.figure(figsize=(25, 14))

# Subplot 1: Sales with hue based on both Year and Category
ax = plt.subplot(2, 1, 1)
sns.barplot(x='Sub-Category', y='Sales',hue='Year' ,data=df, errorbar=None)

# Subplot 2: Margin with hue based on both Year and Category
ax = plt.subplot(2, 1, 2)
sns.barplot(x='Sub-Category', y='Margin', hue= 'Year',  data=df, errorbar=None)

plt.show()

df['Days to Ship'].value_counts()

plt.figure(figsize=(25, 14))

ax = plt.subplot(2, 1, 1)
sns.barplot(x='Quantity', y='Actual Discount',hue='Category',data=df, errorbar=None)

ax = plt.subplot(2, 1, 2)
sns.barplot(x='Quantity', y='Margin',hue='Sub-Category',data=df[df['Category'] == 'Furniture'], errorbar=None)
plt.show()

df.head()

df['Sub-Category'].value_counts()

sns.barplot(x='Days to Ship', y='Profit',hue='Category' , data=df, errorbar=None)

plt.show()
#corr=df.corr()

#sns.heatmap(corr,annot= True)

df.drop(['Margin','Year','Month','Customer Name','Order ID'],axis=1,inplace= True)

dfif= df.select_dtypes(include=['int','float'])

dfif.dtypes

x=0
plt.figure(figsize=(25,25))
for i in dfif.columns:
  ax= plt.subplot(321+x)
  sns.boxplot(dfif[i])
  plt.title(i)
  x+=1

df.describe(percentiles=[0.005,0.01,0.02,0.03,0.95,0.98,0.99,0.996,0.997,0.998,0.999]).T

print(df[df['Sales']>3220.8680].shape)
print(df[df['Profit']>1109.6240].shape)
print(df[df['Profit']<-626.34].shape)

df['Sales'] = np.where(df['Sales'] > 3220.8680,3220.8680, df['Sales'])
df['Profit'] = np.where(df['Profit']>1109.6240, 1109.6240, df['Profit'])
df['Profit'] = np.where(df['Profit']<-626.34, -626.34, df['Profit'])

df.describe(percentiles=[0.005,0.01,0.02,0.03,0.95,0.98,0.99,0.996,0.997,0.998,0.999]).T

df.columns

from sklearn.model_selection import train_test_split

df['State'].unique()

list(df['State'].value_counts())

value_count= df['State'].value_counts()
value_count= value_count[value_count<=10]
value_count

rows_to_drop=list(value_count.index)
rows_to_drop

df.drop(df[df['State'].isin(rows_to_drop)].index,axis=0,inplace= True)

value=df['City'].value_counts()
value=value[value<5]
value

rows_to_drop1=list(value.index)
list(rows_to_drop1)

df['Product']= df['Product Name'].str.split().str[0]

df.drop(df[df['City'].isin(rows_to_drop1)].index,axis=0,inplace= True)

df.dtypes

df.drop(['Product Name','Order Date','Ship Date'],axis=1,inplace= True)

x= df.drop('Sales',axis=1)
y= df['Sales']

xtrain,xtest,ytrain,ytest= train_test_split(x,y,random_state=25,test_size=0.20)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

d= {'model':[],'mse':[],'rmse':[],'mae':[],'r2s':[]}
def model_eval(model,ytest,ypred):
  mse= mean_squared_error(ytest,ypred)
  mae= mean_absolute_error(ytest,ypred)
  rmse= np.sqrt(mse)
  r2s= r2_score(ytest,ypred)
  print('mse:', mse)
  print('rmse:', rmse)
  print('r2s:', r2s)
  print('mae:', mae)
  d['model'].append(model)
  d['mse'].append(mse)
  d['rmse'].append(rmse)
  d['mae'].append(mae)
  d['r2s'].append(r2s)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df.head()

print(xtrain.dtypes)

step1= ColumnTransformer(transformers=[('ohe',OneHotEncoder(drop='first',sparse=False),[4,5,6,7,8,9,10,11,13])],
                         remainder='passthrough')

step2 = LinearRegression()
pipelr= Pipeline([('step1',step1),('step2',step2)])

pipelr.fit(xtrain,ytrain)
ypredlr=pipelr.predict(xtest)

model_eval('regression',ytest,ypredlr)

# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [4,5,6,7,8,9,10,11,13])],
                          remainder='passthrough')

# Step 2: Ridge Regression model with alpha=2.1
step2 = Ridge(alpha=2.1)

# Create a pipeline with the defined steps
piperid = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline to the training data
piperid.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredrid = piperid.predict(xtest)

# Evaluate the performance of the Ridge Regression model
model_eval('Ridge', ytest, ypredrid)
# This step calls the 'model_eval' function to evaluate the performance of the Ridge Regression model.

step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', sparse=False), [4,5,6,7,8,9,10,11,13])],
                          remainder='passthrough')
step2 = RandomForestRegressor(n_estimators=48, max_depth=12, min_samples_split=17, random_state=14)

# Create the pipeline by combining the preprocessing steps and the model
piperf = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline on the training data
piperf.fit(xtrain, ytrain)

# Make predictions on the test data
ypredrf = piperf.predict(xtest)

# Evaluate the model using the model_eval function
model_eval('rf', ytest, ypredrf)

from sklearn.ensemble import BaggingRegressor
# Define the preprocessing steps and the BaggingRegressor with DecisionTreeRegressor base estimator
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False), [4,5,6,7,8,9,10,11,13])],
                          remainder='passthrough')
step2 = BaggingRegressor(base_estimator=RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=9, random_state=5),
                         n_estimators=15, max_samples=xtrain.shape[0], max_features=xtrain.shape[1], random_state=2022)

# Create the pipeline by combining the preprocessing steps and the model
pipebrdt = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline on the training data
pipebrdt.fit(xtrain, ytrain)

# Make predictions on the test data
ypredbrdt = pipebrdt.predict(xtest)

# Evaluate the model using the model_eval function
model_eval('bgdt', ytest, ypredbrdt)

step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False),[4,5,6,7,8,9,10,11,13])],
                          remainder='passthrough')

# Step 2: AdaBoost Regression with RandomForestRegressor as the base estimator
step2 = AdaBoostRegressor(RandomForestRegressor(n_estimators=48, max_depth=12, min_samples_split=17, random_state=14),
                          n_estimators=11)


# Create a pipeline with the defined steps
pipeadar = Pipeline([('step1', step1), ('step2', step2)])
# This pipeline combines the ColumnTransformer and AdaBoost Regression into a single object.

# Fit the pipeline to the training data
pipeadar.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredadar = pipeadar.predict(xtest)

# Evaluate the performance of the AdaBoost Regression with RandomForest model
model_eval('adarf', ytest, ypredadar)

step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first', sparse=False), [4,5,6,7,8,9,10,11,13])],
                          remainder='passthrough')
step2 = DecisionTreeRegressor(max_depth=13, min_samples_split=9, random_state=5)

# Create the pipeline by combining the preprocessing steps and the model
pipedt = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline on the training data
pipedt.fit(xtrain, ytrain)

# Make predictions on the test data
ypreddt = pipedt.predict(xtest)

# Evaluate the model using the model_eval function
model_eval('dt', ytest, ypreddt)

from sklearn.ensemble import GradientBoostingRegressor
# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first',handle_unknown='ignore', sparse=False), [4,5,6,7,8,9,10,11,13])],
                          remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=10,  max_depth=55, random_state=42)

# Create a pipeline with the defined steps
pipegb = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline to the training data
pipegb.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredgb = pipegb.predict(xtest)

# Evaluate the performance of the Gradient Boosting Regression model
model_eval('gb', ytest, ypredgb)

from sklearn.linear_model import HuberRegressor
# Step 1: ColumnTransformer for categorical feature encoding and passthrough
step1 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(drop='first',handle_unknown='ignore', sparse=False), [4,5,6,7,8,9,10,11,13])],
                          remainder='passthrough')

step2 = HuberRegressor(epsilon=5,max_iter=10,alpha=1)

# Create a pipeline with the defined steps
pipehr = Pipeline([('step1', step1), ('step2', step2)])

# Fit the pipeline to the training data
pipehr.fit(xtrain, ytrain)

# Predict the target variable using the pipeline
ypredgb = pipehr.predict(xtest)

# Evaluate the performance of the Gradient Boosting Regression model
model_eval('hr', ytest, ypredgb)

d



data= pd.DataFrame(d)

data

import pickle

pickle.dump(pipeadar,open('pipeadr.pkl','wb'))
pickle.dump(df,open('ordermain.pkl','wb'))

model=pickle.load(open('pipeadr.pkl','rb'))

type(model)

random20= df.sample(20)

random20

prediction= model.predict(random20)

random20['prediction']= prediction

random20.head()

random20=random20[['Profit','prediction']]

random20


