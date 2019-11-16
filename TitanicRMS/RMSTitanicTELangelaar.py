# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:03:13 2019

@author: Thijme Langelaar
"""
import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# %% Data Import
fn_train = 'train.csv'
pn_train = '/home/thijme/Documents/Gitrepos/Kaggle/TitanicRMS/'
fn_test = 'test.csv'
pn_test = '/home/thijme/Documents/Gitrepos/Kaggle/TitanicRMS/'
fn_submit = 'stackedsubmission.csv'
pn_submit = '/home/thijme/Documents/Gitrepos/Kaggle/TitanicRMS/'

df_train = pd.read_csv(pn_train + fn_train, index_col=0, header=0)
df_test = pd.read_csv(pn_test + fn_test, index_col=0, header=0)

# %% Functions & Classes definitions

def title_names(name):
	title = re.split(',',name)[1]
	title = re.search(r'\w+\.',title)
	return title.group()


class SklearnHelper(object):
	def __init__(self, clf, seed=0, params=None):
		if clf is not KNeighborsClassifier:
			params['random_state'] = seed
		
		self.clf = clf(**params)
	
	def train(self, X_train, y_train):
		self.clf.fit(X_train, y_train)
	
	def predict(self, X):
		return self.clf.predict(X)
	
	def fit(self, X, y):
		return self.clf.fit(X, y)
	
	def score(self, X, y):
		return self.clf.score(X, y)
	
	def feature_importances(self, X, y):
		print(self.clf.fit(X,y).feature_importances_)
		return self.clf.fit(X,y).feature_importances_


# %% Data cleaning

full_data = [df_train, df_test]
for dataset in full_data:
	# make Pclass category
	cat_dtype = pd.api.types.CategoricalDtype(categories=[1, 2, 3], ordered=True)
	dataset.Pclass = dataset.Pclass.astype(cat_dtype)
	# 'Sex' column --> 'Male', OHE
	dataset.rename(columns={'Sex': 'Male'}, inplace=True)
	dataset.Male.replace({'male': 1, 'female':0}, inplace=True)
	# make 'Title' column
	dataset['Title'] = dataset['Name'].apply(title_names)
	dataset.Title.replace(['Don.', 'Dona.' ,'Rev.','Dr.','Major.','Lady.',
							'Sir.', 'Col.','Capt.','Countess.','Jonkheer.'],
							'Rare', inplace=True)
	dataset.Title.replace('Mme.','Mrs.', inplace=True)
	dataset.Title.replace('Mlle.','Miss.', inplace=True)
	dataset.Title.replace('Ms.','Miss.', inplace=True)
	title_mapping = {'Mr.': 1, 'Mrs.': 2, 'Miss.': 3, 'Master.': 4, 'Rare':5}
	dataset.Title = dataset.Title.map(title_mapping)
	# make 'Adult' column
	dataset.loc[dataset['Title']== 4, 'Adult'] = 0 #title for a boy
	dataset.loc[dataset['Title']== 1, 'Adult'] = 1
	dataset.loc[dataset['Title']== 2, 'Adult'] = 1
	dataset.loc[dataset['Title']== 3, 'Adult'] = 0
	dataset.loc[dataset['Title']== 5, 'Adult'] = 1
	mask_kid = dataset.Age < 18
	mask_adult = dataset.Age >= 18
	dataset.loc[mask_adult, 'Adult'] = 1
	dataset.loc[mask_kid, 'Adult'] = 0
	dataset.Adult = dataset.Adult.astype(int)
	
	
# fixing missing fare values
for dataset in full_data:
	fare_median = dataset.Fare.median()
	fare_mean = dataset.Fare.mean()
	# OHE Fare
	dataset['Fare'] = dataset['Fare'].fillna(0)
	dataset.loc[dataset['Fare']<= fare_median, 'Fare'] = 1
	dataset.loc[(dataset['Fare']> fare_median) & (dataset['Fare']<= fare_mean), 'Fare'] = 2
	dataset.loc[(dataset['Fare']> fare_mean), 'Fare'] = 3
	dataset.Fare = dataset.Fare.astype(int)

# fixing missing values age
for dataset in full_data:
	kid_age_mean = int(np.round(dataset.loc[dataset['Adult']==0,'Age'].mean()))
	adult_age_mean = int(np.round(dataset.loc[dataset['Adult']==1,'Age'].mean()))
	dataset.loc[dataset['Adult']==0,'Age'] = dataset.loc[dataset['Adult']==0,'Age'].fillna(kid_age_mean)
	dataset.loc[dataset['Adult']==1,'Age'] = dataset.loc[dataset['Adult']==1,'Age'].fillna(adult_age_mean)

# %% Feature construction
for dataset in full_data:
	dataset['Embarked'] = dataset['Embarked'].fillna('S')
	dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
	dataset['Cabin'] = dataset['Cabin'].fillna('Unknown')
	dataset['Cabin'] = dataset['Cabin'].str[0]
	dataset['Has_Cabin'] = 1
	dataset.loc[dataset['Cabin']=='U', 'Has_Cabin'] = 0
	dataset['Family_Size'] = dataset['Parch'] + dataset['SibSp'] + 1
	dataset['Is_Alone'] = 0
	dataset.loc[dataset['Family_Size']==1, 'Is_Alone'] = 1

# %% EDA
#sns.heatmap(df_train.corr(),annot=True)
#plt.show()
	
print(df_train.groupby('Pclass').Cabin.value_counts())

plt.figure()
sns.countplot(x='Pclass', hue='Male', data=df_train)
plt.show()

tmpPclass0 = df_train.Pclass[df_train['Survived'] == 0]
tmpMale0 = df_train.Male[df_train['Survived'] == 0]
tmpPclass1 = df_train.Pclass[df_train['Survived'] == 1]
tmpMale1 = df_train.Male[df_train['Survived'] == 1]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
sns.countplot(x='Pclass', hue='Male', data=df_train, ax=ax1)
ax1.title.set_text('Distribution')

sns.countplot(x=tmpPclass1, hue=tmpMale1, ax=ax2)
ax2.title.set_text('Survived')

sns.countplot(x=tmpPclass0, hue=tmpMale0, ax=ax3)
ax3.title.set_text('Died')
plt.show()

plt.figure()
sns.countplot(x='Pclass', hue='Adult', data=df_train)
plt.show()


# %% Machine Learning
features = ['Pclass', 'Male', 'Adult', 'Embarked', 'Fare', 'Title',
			'Is_Alone', 'Has_Cabin']

# Some useful parameters which will come in handy later on
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
	}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
	}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
	}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
	}

# KNearest Neighbor Classifier parameters
knn_params = {
	'n_jobs' : -1,
	'n_neighbors' : 5
	}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
knn = SklearnHelper(clf=KNeighborsClassifier, seed=SEED, params=knn_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

models = [rf, et, ada, gb, knn, svc]
modelnames = ['Random Forest', 'Extra Trees', 'AdaBoost', 'GradientBoost',
			  'KNearestNeighbor', 'SupportVectorMachine']

X = df_train[features].values
X_real = df_test[features].values
y = df_train['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
y_pred_test = pd.DataFrame()
y_pred_real = pd.DataFrame()
i = 0
for model in models:
	modelname = modelnames[i]
	model.train(X_train, y_train)
	y_pred_test[modelname] = model.predict(X_test)
	y_pred_real[modelname] = model.predict(X_real)
	score = model.score(X_test, y_test)
	print('Model: ' + modelname + ' ' + str(score))
	i += 1
	

rf_feature = rf.feature_importances(X_train, y_train)
et_feature = et.feature_importances(X_train, y_train)
ada_feature = ada.feature_importances(X_train, y_train)
gb_feature = gb.feature_importances(X_train, y_train)

feature_dataframe = pd.DataFrame({'Features': features,
								 'Random Forest': rf_feature,
								 'Extra Trees': et_feature,
								 'AdaBoost': ada_feature,
								 'GradientBoost': gb_feature
								  })
feature_dataframe['mean'] = feature_dataframe.mean(axis=1)

# %% 
X_train = y_pred_test.values
X_real = y_pred_real.values

# %%			   
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(X_train, y_test)
predictions = gbm.predict(X_real)


# %% Data Export
submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission['PassengerId'] = df_test.index
submission['Survived'] = predictions

print(submission.info())

submission.to_csv(pn_submit + fn_submit, columns=['PassengerId','Survived'], index=False)

