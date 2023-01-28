#!/usr/bin/env python
# coding: utf-8

# In[54]:


from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)


# In[55]:


#classification comparison linear, decision tree, random forest, svm, boosting, neural network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


# In[92]:


df = pd.read_csv('Downloads/used_cars_data.csv', sep = ',')
df2 = pd.read_csv('Downloads/used_cars_data.csv', sep = ',')


# In[93]:


df.head()


# In[94]:


df.describe()


# In[95]:


df.isnull().sum()


# In[96]:


df['brand_average'] = df.groupby(['brand'])['price (eur)'].transform('mean').round(2)
df['brand_average']


# In[97]:


def group_brand(value):
    if value > 20000:
        return 0
    elif value > 15000 and value < 20000:
        return 1
    elif value > 10000 and value < 15000:
        return 2
    else:
        return 3
df['brand_group'] = df.apply(lambda x: group_brand(x['brand_average']),axis=1)
df = df.drop('brand', axis = 1)
df = df.drop('brand_average', axis = 1)
df.head()


# In[98]:


df['model_average'] = df.groupby(['model'])['price (eur)'].transform('mean').round(2)
df['model_average']


# In[99]:


def group_model(value):
    if value > 30000:
        return 0
    elif value > 20000 and value < 30000:
        return 1
    elif value > 15000 and value < 20000:
        return 2
    elif value > 10000 and value < 15000:
        return 3
    else:
        return 4
df['model_group'] = df.apply(lambda x: group_model(x['model_average']),axis=1)
df = df.drop('model', axis = 1)
df = df.drop('model_average', axis = 1)
df.head()


# In[100]:


df['year'].value_counts()


# In[101]:


#few values in the years up to 2010, so let's remove these values
df = df[df['year'] > 4]
sns.boxplot(x="year", y="price (eur)", data=df)


# In[102]:


def year_model(value):
    if value > 2017:
        return 0
    elif value > 2012 and value < 2017:
        return 1
    elif value > 2006 and value < 2012:
        return 2
    else:
        return 3
df['year_group'] = df.apply(lambda x: year_model(x['year']),axis=1)
df = df.drop('year', axis = 1)
df.head()


# In[103]:


#group engine variable 
df['engine_average'] = df.groupby(['engine'])['price (eur)'].transform('mean').round(2)
def group_engine(value):
    if value > 40000:
        return 0
    elif value > 30000 and value < 40000:
        return 1
    elif value > 20000 and value < 30000:
        return 2
    elif value > 10000 and value < 20000:
        return 3
    else:
        return 4
df['engine_group'] = df.apply(lambda x: group_engine(x['engine_average']),axis=1)
df = df.drop('engine', axis = 1)
df = df.drop('engine_average', axis = 1)
df.head()


# In[104]:


correl = df.corr().round(2)
plt.figure(figsize = (15, 10))
sns.heatmap(correl, annot = True)


# In[105]:



engine = df2[['engine','Unnamed: 0']].groupby('engine').agg('count').sort_values('Unnamed: 0', ascending=False).reset_index()
engine


# In[106]:


brand = df2[['brand','Unnamed: 0']].groupby('brand').agg('count').sort_values('Unnamed: 0', ascending=False).reset_index()
brand


# In[107]:


model = df2[['model','Unnamed: 0']].groupby('model').agg('count').sort_values('Unnamed: 0', ascending=False).reset_index()
model.head()


# In[108]:


location = df2[['location','Unnamed: 0']].groupby('location').agg('count').sort_values('Unnamed: 0', ascending=False).reset_index()
location.head()


# In[109]:



plt.subplot(1,2,1)
plt.gca().set_title('Variable Fuel')
sns.countplot(x = 'fuel', palette = 'Set2', data = df)

plt.subplot(1,2,2)
plt.gca().set_title('Variable gearbox')
sns.countplot(x = 'gearbox', palette = 'Set2', data = df)


# In[113]:



sns.countplot(x = 'year', palette = 'Set2', data = df2)


# In[114]:


plt.figure(figsize = (15,5))
sns.barplot(data=brand.head(10), x="Unnamed: 0", y="brand")


# In[115]:


plt.figure(figsize = (15,5))
sns.barplot(data=model.head(10), x="Unnamed: 0", y="model")


# In[116]:


plt.figure(figsize = (15,5))
sns.barplot(data=engine.head(10), x="Unnamed: 0", y="engine")


# In[117]:


plt.figure(figsize = (15,5))
sns.barplot(data=location.head(10), x="Unnamed: 0", y="location")


# In[118]:


sns.scatterplot(data=df, x="mileage (kms)", y="price (eur)", palette = 'Set2')


# In[119]:


df = df.drop('Unnamed: 0', axis = 1)

location = pd.get_dummies(df['location'])
df = pd.concat([df, location], axis = 1)
df.head()


# In[120]:


df = df.drop(['location'], axis = 1)


# In[121]:


from sklearn.preprocessing import LabelEncoder

label_encoder_fuel = LabelEncoder()
label_encoder_gearbox = LabelEncoder()

df['fuel'] = label_encoder_fuel.fit_transform(df['fuel'])
df['gearbox'] = label_encoder_gearbox.fit_transform(df['gearbox'])
df['fuel'] 


# In[197]:


X = df.drop('price (eur)', axis = 1)


# In[198]:


X = X.values


# In[199]:


y = df['price (eur)']


# In[200]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
y_standard = scaler.fit_transform(y.values.reshape(-1,1))


# In[201]:


from sklearn.model_selection import train_test_split
y=y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[202]:


#learn regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_normal_score_train = lr_model.score(X_train, y_train)
lr_normal_score_test = lr_model.score(X_test, y_test)
previsoes = lr_model.predict(X_test)
mae_lr_normal = mean_absolute_error(y_test, previsoes)
rmse_lr_normal = np.sqrt(mean_squared_error(y_test, previsoes))

print('Train :', lr_normal_score_train)
print('Test :', lr_normal_score_test)
print('Mean Absolute Error :', mae_lr_normal)
print('Root Mean Square Error :', rmse_lr_normal)


# In[203]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
lr_poly = LinearRegression()
lr_poly.fit(X_poly_train, y_train)
lr_poly_normal_score_train = lr_poly.score(X_poly_train, y_train)
lr_poly_normal_score_test = lr_poly.score(X_poly_test, y_test)
previsoes = lr_poly.predict(X_poly_test)
mae_poly_normal = mean_absolute_error(y_test, previsoes)
rmse_poly_normal = np.sqrt(mean_squared_error(y_test, previsoes))

print('Train :', lr_poly_normal_score_train)
print('Test :', lr_poly_normal_score_test)
print('Mean Absolute Error :', mae_poly_normal)
print('Root Mean Square Error :', rmse_poly_normal)


# In[204]:


from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree = 3)
X_poly_train2 = poly2.fit_transform(X_train)
X_poly_test2 = poly2.transform(X_test)
lr_poly2 = LinearRegression()
lr_poly2.fit(X_poly_train2, y_train)
lr_poly_normal_score_train2 = lr_poly2.score(X_poly_train2, y_train)
lr_poly_normal_score_test2 = lr_poly2.score(X_poly_test2, y_test)
previsoes2 = lr_poly2.predict(X_poly_test2)
mae_poly_normal2 = mean_absolute_error(y_test, previsoes2)
rmse_poly_normal2 = np.sqrt(mean_squared_error(y_test, previsoes2))

print('Train :', lr_poly_normal_score_train2)
print('Test :', lr_poly_normal_score_test2)
print('Mean Absolute Error :', mae_poly_normal2)
print('Root Mean Square Error :', rmse_poly_normal2)


# In[205]:


#decision tree parameter selection
from sklearn.tree import DecisionTreeRegressor

min_split = np.array([2, 3, 4, 5, 6, 7])
max_nvl = np.array([3, 4, 5, 6, 7, 9, 11])
alg = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
values_grid = {'min_samples_split': min_split, 'max_depth': max_nvl, 'criterion': alg}

model = DecisionTreeRegressor()
gridDecisionTree = GridSearchCV(estimator = model, param_grid = values_grid, cv = 5, n_jobs = -1)
gridDecisionTree.fit(X_train, y_train)
print('Mín Split: ', gridDecisionTree.best_estimator_.min_samples_split)
print('Max Nvl: ', gridDecisionTree.best_estimator_.max_depth)
print('Algorithm: ', gridDecisionTree.best_estimator_.criterion)
print('Score: ', gridDecisionTree.best_score_)


# In[206]:



decision_tree = DecisionTreeRegressor(min_samples_split = 3, max_depth = 4, criterion = 'squared_error')
decision_tree.fit(X_train, y_train)
lr_normal_decision_tree = decision_tree.score(X_train, y_train)
lr_normal_decision_tree_test = decision_tree.score(X_test, y_test)
previsoes = decision_tree.predict(X_test)
mae_lr_normal_decision_tree  = mean_absolute_error(y_test, previsoes)
rmse_lr_normal_decision_tree = np.sqrt(mean_squared_error(y_test, previsoes))

print('Train :', lr_normal_decision_tree)
print('Test :', lr_normal_decision_tree_test)
print('Mean Absolute Error :', mae_lr_normal_decision_tree)
print('Root Mean Square Error :', rmse_lr_normal_decision_tree)


# In[207]:


#feature selection
columns = df.drop('price (eur)', axis = 1).columns
feature_imp = pd.Series(decision_tree.feature_importances_, index = columns).sort_values(ascending = False)
feature_imp


# In[208]:


#random forest
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 100)
regressor_rf.fit(X_train, y_train)
lr_normal_rf = regressor_rf.score(X_train, y_train)
lr_normal_rf_test = regressor_rf.score(X_test, y_test)
previsoes = regressor_rf.predict(X_test)
mae_lr_normal_rf  = mean_absolute_error(y_test, previsoes)
rmse_lr_normal_rf = np.sqrt(mean_squared_error(y_test, previsoes))

print('Train :', lr_normal_rf)
print('Test :', lr_normal_rf_test)
print('Mean Absolute Error :', mae_lr_normal_rf)
print('Root Mean Square Error :', rmse_lr_normal_rf)


# In[209]:


#feature selection
feature_imp_random = pd.Series(regressor_rf.feature_importances_, index = columns).sort_values(ascending = False)
feature_imp_random


# In[244]:


#SVM poly kernel
regressor_svr_poly = SVR(kernel = 'poly', degree = 3)
regressor_svr_poly.fit(X_trainsd, y_trainsd)
standard_svm_poly = regressor_svr_poly.score(X_trainsd, y_trainsd)
standard_svm_poly_test = regressor_svr_poly.score(X_testsd, y_testsd)
previsoes = regressor_svr_poly.predict(X_testsd)
y_test_inverse = scaler.inverse_transform(y_testsd.reshape(-1,1))
previsoes_inverse = scaler.inverse_transform(previsoes.reshape(-1,1))

mae_svr_poly_standard  = mean_absolute_error(y_test_inverse, previsoes_inverse)
rmse_svm_poly_standard = np.sqrt(mean_squared_error(y_test_inverse, previsoes_inverse))

print('Train :', standard_svm_poly)
print('Test :', standard_svm_poly_test)
print('Mean Absolute Error :', mae_svr_poly_standard)
print('Root Mean Square Error :', rmse_svm_poly_standard)


# In[245]:


#RNA
from sklearn.neural_network import MLPRegressor
regressor_rna = MLPRegressor(max_iter = 1000, hidden_layer_sizes=(4, 4))
regressor_rna.fit(X_trainsd, y_trainsd)
standard_rna = regressor_rna.score(X_trainsd, y_trainsd)
standard_rna_test = regressor_rna.score(X_testsd, y_testsd)
previsoes = regressor_rna.predict(X_testsd)
y_test_inverse = scaler.inverse_transform(y_testsd.reshape(-1,1))
previsoes_inverse = scaler.inverse_transform(previsoes.reshape(-1,1))
mae_rna_standard = mean_absolute_error(y_test_inverse, previsoes_inverse)
rmse_rna_standard = np.sqrt(mean_squared_error(y_test_inverse, previsoes_inverse))

print('Train :', standard_rna)
print('Test :', standard_rna_test)
print('Mean Absolute Error :', mae_rna_standard)
print('Root Mean Square Error :', rmse_rna_standard)


# In[255]:


#adaboost
from sklearn.ensemble import AdaBoostRegressor
n_estimators = np.array([500])
learning_rate = np.array([1.0, 1.1, 0.01, 0.2, 0.3, 0.4])
values_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
model = AdaBoostRegressor()
gridAdaBoost = GridSearchCV(estimator = model, param_grid = values_grid, cv = 5, n_jobs = 1)
gridAdaBoost.fit(X_train, y_train)
print('Learning Rate: ', gridAdaBoost.best_estimator_.learning_rate)
print('Score: ', gridAdaBoost.best_score_)


# In[256]:



ada_boost = AdaBoostRegressor(learning_rate = 1.1, n_estimators = 500)
ada_boost.fit(X_trainsd, y_trainsd)
lr_normal_ada_boost = ada_boost.score(X_trainsd, y_trainsd)
lr_normal_ada_boost_test = ada_boost.score(X_testsd, y_testsd)
previsoes = ada_boost.predict(X_testsd)
y_test_inverse = scaler.inverse_transform(y_testsd.reshape(-1,1))
previsoes_inverse = scaler.inverse_transform(previsoes.reshape(-1,1))
mae_lr_normal_ada_boost  = mean_absolute_error(y_test_inverse, previsoes_inverse)
rmse_ada_boost = np.sqrt(mean_squared_error(y_test_inverse, previsoes_inverse))

print('Train :', lr_normal_ada_boost)
print('Test :', lr_normal_ada_boost_test)
print('Mean Absolute Error :', mae_lr_normal_ada_boost)
print('Root Mean Square Error :', rmse_ada_boost)


# In[259]:


#gradient boost
from sklearn.ensemble import GradientBoostingRegressor
n_estimators = np.array([500])
learning_rate = np.array([0.01, 0.02, 0.003, 0.0001, 0.5, 0.4])
criterion = np.array(['friedman_mse', 'squared_error'])
values_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'criterion': criterion}
model = GradientBoostingRegressor()
gridGradientBoost = GridSearchCV(estimator = model, param_grid = values_grid, cv = 5, n_jobs = 1)
gridGradientBoost.fit(X_trainsd, y_trainsd)
print('Learning Rate: ', gridGradientBoost.best_estimator_.learning_rate)
print('Criterion: ', gridGradientBoost.best_estimator_.criterion)
print('Score: ', gridGradientBoost.best_score_)


# In[260]:


grad_boost = GradientBoostingRegressor(learning_rate = 0.02, n_estimators = 500, criterion = 'friedman_mse')
grad_boost.fit(X_trainsd, y_trainsd)
lr_normal_grad_boost = grad_boost.score(X_trainsd, y_trainsd)
lr_normal_grad_boost_test = grad_boost.score(X_testsd, y_testsd)
previsoes = grad_boost.predict(X_testsd)
y_test_inverse = scaler.inverse_transform(y_testsd.reshape(-1,1))
previsoes_inverse = scaler.inverse_transform(previsoes.reshape(-1,1))
mae_lr_normal_grad_boost  = mean_absolute_error(y_test_inverse, previsoes_inverse)
rmse_grad_boost = np.sqrt(mean_squared_error(y_test_inverse, previsoes_inverse))

print('Train :', lr_normal_grad_boost)
print('Test :', lr_normal_grad_boost_test)
print('Mean Absolute Error :', mae_lr_normal_grad_boost)
print('Root Mean Square Error :', rmse_grad_boost)


# In[261]:


#feature selection by gradient boost 
feature_imp_grad = pd.Series(grad_boost.feature_importances_, index = columns).sort_values(ascending = False)
feature_imp_grad


# In[264]:


linear_regression = {'Model':'Linear Regression',
               'Score Train':lr_normal_score_train,
               'Score Test':lr_normal_score_test,
               'MSE':mae_lr_normal,
               'RMSE':rmse_lr_normal,}

polynomial_regression = {'Model':'Polynomial Regression',
               'Score Train':lr_poly_normal_score_train,
               'Score Test':lr_normal_score_test,
               'MSE':mae_poly_normal,
               'RMSE':rmse_poly_normal,}

decision_tree = {'Model':'Decision Tree',
               'Score Train':lr_normal_decision_tree,
               'Score Test':lr_normal_decision_tree_test,
               'MSE':mae_lr_normal_decision_tree,
               'RMSE':rmse_lr_normal_decision_tree,}

random_forest = {'Model':'Random Forest',
               'Score Train':lr_normal_rf,
               'Score Test':lr_normal_rf_test,
               'MSE':mae_lr_normal_rf,
               'RMSE':rmse_lr_normal_rf,}

svr_polynomial = {'Model':'SVR Polynomial',
               'Score Train':standard_svm_poly,
               'Score Test':standard_svm_poly_test,
               'MSE':mae_svr_poly_standard,
               'RMSE':rmse_svm_poly_standard,}

rna = {'Model':'RNA',
               'Score Train':standard_rna,
               'Score Test':standard_rna_test,
               'MSE':mae_rna_standard,
               'RMSE':rmse_rna_standard,}

ada = {'Model':'Ada Boost',
               'Score Train':lr_normal_ada_boost,
               'Score Test':lr_normal_ada_boost_test,
               'MSE':mae_lr_normal_ada_boost,
               'RMSE':rmse_ada_boost,}

grad = {'Model':'Gradient Boosting',
               'Score Train':lr_normal_grad_boost,
               'Score Test':lr_normal_grad_boost_test,
               'MSE':mae_lr_normal_grad_boost,
               'RMSE':rmse_grad_boost,}


resume = pd.DataFrame({'Linear Regression':pd.Series(linear_regression),
                       'Polynomial Regression':pd.Series(polynomial_regression),
                       'Decision Tree':pd.Series(decision_tree),
                       'Random Forest':pd.Series(random_forest),
                       'SVR Polynomial':pd.Series(svr_polynomial),
                       'RNA':pd.Series(rna),
                       'AdaBoost':pd.Series(ada),
                       'GradientBoosting':pd.Series(grad)
                      })


# In[265]:


resume


# In[ ]:




