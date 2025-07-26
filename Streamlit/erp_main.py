# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:33:25 2025

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

train=pd.read_csv('aug_train.csv')
test=pd.read_csv('aug_test.csv')

categorical_col=["gender","enrolled_university","education_level","major_discipline","company_size","last_new_job"]
for col in categorical_col:
    train[col].fillna(train[col].mode()[0],inplace=True)
    test[col].fillna(test[col].mode()[0],inplace=True)
    
def convert_experience(value):
    if value=="<1":
        return 0
    elif value==">20":
        return 21
    elif pd.isna(value):
        return None
    else:
        return int(value)

train["experience"]=train["experience"].apply(convert_experience)
test["experience"]=test["experience"].apply(convert_experience)

train["experience"].fillna(train["experience"].median(),inplace=True)
test["experience"].fillna(test["experience"].median(),inplace=True)

def convert_company_size(size):
    if pd.isna(size):
        return np.nan
    elif size == '<10':
        return 5
    elif size == '10-49':
        return 30
    elif size == '50-99':
        return 75
    elif size == '100-500':
        return 300
    elif size == '500-999':
        return  750
    elif size == '1000-4999':
        return 2500
    elif size == '5000-9999':
        return 7500
    elif size == '10000+':
        return 10000
    else:
        return np.nan

train['company_size']=train['company_size'].apply(convert_company_size)
test['company_size']=test['company_size'].apply(convert_company_size)  

train["company_size"].fillna(train["company_size"].median(),inplace=True)
test["company_size"].fillna(test["company_size"].median(),inplace=True)

from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()

train['gender']=encode.fit_transform(train['gender'])
test['gender']=encode.fit_transform(test['gender'])

train['city']=encode.fit_transform(train['city'])
test['city']=encode.fit_transform(test['city'])
  
train['relevent_experience']=encode.fit_transform(train['relevent_experience'])
test['relevent_experience']=encode.fit_transform(test['relevent_experience'])

train.enrolled_university.unique()

train['enrolled_university']=encode.fit_transform(train['enrolled_university'])
test['enrolled_university']=encode.fit_transform(test['enrolled_university'])

train['education_level']=encode.fit_transform(train['education_level'])
test['education_level']=encode.fit_transform(test['education_level'])

train['major_discipline']=encode.fit_transform(train['major_discipline'])
test['major_discipline']=encode.fit_transform(test['major_discipline'])

train['company_type']=encode.fit_transform(train['company_type'])
test['company_type']=encode.fit_transform(test['company_type'])

mapping = {
    'never': 0,  # No previous job
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '>4': 5  # More than 4 years
}

train['last_new_job'] = train['last_new_job'].replace(mapping).astype(int)
test['last_new_job'] = test['last_new_job'].replace(mapping).astype(int)


df_train=train.drop(columns=['enrollee_id','city'],axis=1)
df_test=test.drop(columns=['enrollee_id','city'],axis=1)

# # 4. Separate features and target
X= train.drop(columns=['training_hours', 'target'])
y=train['target']
X_test= test.drop(columns=['training_hours'])

## # 5. Handle class imbalance with SMOTE
smote= SMOTE(random_state=42)
X_resampled, y_resampled= smote.fit_resample(X,y)

# 6. Split data
X_train, X_val, y_train, y_val= train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 7. Scale features
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_val_scaled= scaler.transform(X_val)
X_test_scaled= scaler.transform(X_test)

## LIGHTGBM model
model4= LGBMClassifier(random_state=42)

model4.fit(X_train_scaled, y_train)

y_pred= model4.predict(X_val_scaled)
acc= accuracy_score(y_val, y_pred)
roc_auc= roc_auc_score(y_val, y_pred)

print(acc)
print(roc_auc)

## save the model
joblib.dump(model4,"lightgbm_model.joblib")
joblib.dump(scaler,"standardscaler_scaler.joblib")































X_train = train.drop(columns=['target'])  # Assuming 'target' is the label
y_train = train['target']

X_test = test.copy()  # No target in test data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Check class distribution after SMOTE
print(pd.Series(y_train_resampled).value_counts())

from xgboost import XGBClassifier
xgb=XGBClassifier()

xgb.fit(X_train_resampled,y_train_resampled)

y_pred=xgb.predict(X_test_scaled)

from sklearn.metrics import classification_report

y_train_pred=xgb.predict(X_train_scaled)
print(classification_report(X_train_resampled,y_train_resampled))


