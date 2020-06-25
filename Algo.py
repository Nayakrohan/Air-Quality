# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:53:13 2020

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/Real-data/Real_Combine.csv')

#Check for null values
#df.isnull()
sns.heatmap(df.isnull(), yticklabels= False, cbar = True, cmap = 'viridis')
df = df.dropna()

X = df.iloc[:, :-1]
y= df.iloc[:,-1]

sns.pairplot(df)

df.corr()

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#Plot Heat Map
#g = sns.heatmap(df.corr(), annot = True, cmap = 'RdYlGn')
g = sns.heatmap(df[top_corr_features].corr(), annot = True, cmap = 'RdYlGn')

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)

#Plot graph of Featurre Importance for better Visualization
feat_importances = pd.Series(model.feature_importances_, index = X.columns)
feat_importances.nlargest(5).plot(kind = 'barh')
plt.show()

sns.distplot(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Coefficient of determination R^2 <-- on train set {}".format(regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on train set {}".format(regressor.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor, X, y, cv = 5)
score.mean()

#Getting slope wrt each independent Variable
regressor.coef_

#Getting y intercept
regressor.intercept_

#Model Evaluation
# (
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns = ['Coefficient'])
print(coeff_df)
#This basically Means
#Holding all other features fixed, a 1 unit increase in T is associated with  an "decrease of 17.24 in AQI PM2.5"
#and so on...

prediction = regressor.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test, prediction)

from sklearn import metrics
print("Mean Absolute Error(MAE): ", metrics.mean_absolute_error(y_test, prediction))
print("Mean Squared Error(MSE): ", metrics.mean_squared_error(y_test, prediction))
print("Root Mean Squared Error(RMSE): ", np.sqrt(metrics.mean_squared_error(y_test, prediction)))

import pickle
#open a file when you want to store the data
file = open('regression.pkl', 'wb')

#dump information to that file
pickle.dump(regressor,file)

## Comparison of Linear , Ridge and Lasso Regression

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

linear_regressor = LinearRegression()
nmse = cross_val_score(linear_regressor, X, y,scoring='neg_mean_squared_error', cv=5)
print(np.mean(nmse))

from sklearn.model_selection import GridSearchCV
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 40]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring= 'neg_mean_squared_error')
ridge_regressor.fit(X, y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 40]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring= 'neg_mean_squared_error')
lasso_regressor.fit(X, y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

import pickle
#open a file when you want to store the data
file = open('lasso_regressor.pkl', 'wb')

#dump information to that file
pickle.dump(lasso_regressor,file)


#Using Decision Tree Algo
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(criterion = 'mse')
dtree.fit(X_train, y_train)
print("Coefficient of determination R^2 <-- on train set {}".format(dtree.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on train set {}".format(dtree.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
score = cross_val_score(dtree, X, y, cv=5)
score.mean()

from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

features = list(df.columns[:-1])
features 

#import os
#os.environ['Path'] = os.environ['Path']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

dot_data = StringIO()
export_graphviz(dtree, out_file = dot_data, feature_names= features, filled = True, rounded = True) 

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

y_pred = dtree.predict(X_test)
sns.distplot(y_test-y_pred)

## Hyperparameter Tunning
DecisionTreeRegressor()
params = {
        "splitter" : ["best", "random"],
        "max_depth" : [3, 4, 6, 8, 10, 12, 15],
        "min_samples_leaf" : [1, 2, 3, 4, 5],
        "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4],
        "max_features" : ["auto", "log2","sqrt", None],
        "max_leaf_nodes" : [None, 10, 20, 30, 40, 50, 60, 70]
        }
from sklearn.model_selection import GridSearchCV
random_search = GridSearchCV(dtree, param_grid= params, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = 10, verbose = 3)
random_search.fit(X_train, y_train)

random_search.best_params_
random_search.best_score_
prediction = random_search.predict(X_test)
sns.distplot(y_test-prediction)

import pickle
file = open('decision_regression_model.pkl', 'wb')
pickle.dump(random_search, file)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor, X_train, y_train, cv=5)
score.mean()
prediction = regressor.predict(X_test)
sns.distplot(y_test-prediction)
plt.scatter(y_test, prediction)

## Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(start = 5, stop = 30, num = 6)]
print(max_depth)

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf}

print(random_grid)

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter = 100, cv =5, verbose = 2, random_state = 42)

rf_random.fit(X_train, y_train)

rf_random.best_params_

rf_random.best_score_

prediction = rf_random.predict(X_test)

sns.distplot(y_test-prediction)

plt.scatter(y_test, prediction)

import pickle
file = open('random_forest_regressor.pkl', 'wb')
pickle.dump(rf_random, file)

import xgboost as xgb
regressor = xgb.XGBRegressor()
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)
regressor.score(X_test, y_test)

## Hyper Parameter Tunning

xgb_regressor = xgb.XGBRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_depth = [3, 4, 5, 8, 10]

learning_rate = [0.01, 0.1, 0.5, 1]

subsample = [0.7, 0.8, 1]

min_child_weight = [0.7, 1, 1.3, 1.5]

param ={
        'n_estimators' : n_estimators,
        'max_depth' : max_depth,
        'learning_rate' : learning_rate,
        'subsample' : subsample,
        'min_child_weight': min_child_weight}

from sklearn.model_selection import RandomizedSearchCV
xgb_random = RandomizedSearchCV(xgb_regressor, param_distributions = param, scoring = 'neg_mean_squared_error', n_iter = 100, cv = 5, verbose = 2, n_jobs = 1)

xgb_random.fit(X_train, y_train)

xgb_random.best_params_
xgb_random.best_score_

predict = xgb_random.predict(X_test)

sns.distplot(y_test-predict)
plt.scatter(y_test, predict)

import pickle
file = open('xgb_regressor','wb')
pickle.dump(xgb_random, file)


















