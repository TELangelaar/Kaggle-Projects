# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:03:13 2019

@author: Thijme Langelaar
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# %% Data Import
fn_train = 'train.csv'
pn_train = 'X:\\My Desktop\\BigData\\GitRepos\\Kaggle-Projects\\TitanicRMS\\'
fn_test = 'test.csv'
pn_test = 'X:\\My Desktop\\BigData\\GitRepos\\Kaggle-Projects\\TitanicRMS\\'
fn_submit = 'submission.csv'
pn_submit = 'X:\\My Desktop\\BigData\\GitRepos\\Kaggle-Projects\\TitanicRMS\\'

# Training Data
df_train = pd.read_csv(pn_train + fn_train, index_col=0, header=0)
df_train.Embarked = df_train.Embarked.astype('category')
df_train.drop(labels='Cabin', axis=1, inplace=True)

cat_dtype = pd.api.types.CategoricalDtype(categories=[1, 2, 3], ordered=True)
df_train.Pclass = df_train.Pclass.astype(cat_dtype)  # Pclass ordered category

df_train.rename(columns={'Sex': 'Male'}, inplace=True)
df_train.Male.replace({'male': 1, 'female': 0}, inplace=True)

# Drop missing values in Age. Could introduce a systematic bias
df_train.dropna(subset=['Age'], inplace=True)

mask = df_train.Age > 18
column_name = 'Adult'
df_train.loc[mask, column_name] = 1
df_train.Adult.fillna(0, inplace=True)


# Test Data
df_test = pd.read_csv(pn_test + fn_test, index_col=0, header=0)
df_test.Embarked = df_test.Embarked.astype('category')
df_test.drop(labels='Cabin', axis=1, inplace=True)

cat_dtype = pd.api.types.CategoricalDtype(categories=[1, 2, 3], ordered=True)
df_test.Pclass = df_test.Pclass.astype(cat_dtype)  # Pclass ordered category

df_test.rename(columns={'Sex': 'Male'}, inplace=True)
df_test.Male.replace({'male': 1, 'female': 0}, inplace=True)

mask = df_test.Age > 18
column_name = 'Adult'
df_test.loc[mask, column_name] = 1
df_test.Adult.fillna(0, inplace=True)

df_test['Survived'] = np.nan

# %% EDA
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
y = df_train['Survived'].values
X = df_train[['Pclass', 'Male', 'Adult']].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

X_realtest = df_test[['Pclass', 'Male', 'Adult']].values
y_realpred = knn.predict(X_realtest)

# %% Data Export
#first_submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
#first_submission['PassengerId'] = df_test.index
#first_submission['Survived'] = y_realpred
#
#print(first_submission.info())
#
#first_submission.to_csv(pn_submit + fn_submit, columns=['PassengerId','Survived'], index=False)

