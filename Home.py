import pandas as pd
import numpy as np 
import missingno as mno
import statistics as st

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
#Columns Names

col_name = train_data.columns

#Visualising The Column which have the missing value
#mno.bar(train_data,labels=True)
mno.matrix(test_data)

#Columns to drop in Training data

c_drop = ['Alley','PoolQC','Fence','MiscFeature','Id']
train_data.drop(columns=c_drop,inplace=True)

#Columns to drop in Test data

ct_drop = ['Alley','PoolQC','Fence','MiscFeature','Id']
test_data.drop(columns=ct_drop,inplace=True)

#Filling missing values in training set

train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean(),inplace=True)


#Categorical Column and Numerical Column For Training Data

cat_train = [cat for cat in train_data if train_data[cat].dtype=='object']
num_train = [cat for cat in train_data if train_data[cat].dtype=='int64' or train_data[cat].dtype=='float64']

#Categorical Column and Numerical Column For Test Data

cat_test = [ct_test for ct_test in test_data if test_data[ct_test].dtype=='object']
num_test = [ct_test for ct_test in test_data if test_data[ct_test].dtype=='int64' or test_data[ct_test].dtype=='float64']

#Filling Missing values in Training set
for col in cat_train:
    train_data[col] = train_data[col].fillna(st.mode(train_data[col].dropna()))
    
for col in num_train:
    train_data[col] = train_data[col].fillna(st.mode(train_data[col].dropna()))

#Filling Missing values in Test Data

for ct_test in cat_test:
    test_data[ct_test] = test_data[ct_test].fillna(st.mode(test_data[ct_test].dropna()))

for ct_test in num_test:
    test_data[ct_test] = test_data[ct_test].fillna(st.mode(test_data[ct_test].dropna()))

#Doing label encoding in Training set

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_train:
    train_data[col] = le.fit_transform(train_data[col])

#Doing label encoding in Test set

for ct_test in cat_test:
    test_data[ct_test] = le.fit_transform(test_data[ct_test])

#Splitting training data into train and test data

from sklearn.model_selection import train_test_split
X_train = train_data.iloc[: , :-1].values
y_train = train_data.iloc[: , -1].values
X_test = test_data.iloc[: , :].values

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,random_state=0,test_size=0.25)


#Training the model

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)

#Testing the model
y_pred = regressor.predict(X_test).reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Performance metrics
errors = abs(y_pred - y_test)
print('Metrics for DecisionTreeRegressor Trained on Original Data')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 5), '%.')

predictions = regressor.predict(test_data)
predictions.shape

sample['SalePrice'] = predictions.astype(float)

sample.to_csv('Final_submission.csv')
