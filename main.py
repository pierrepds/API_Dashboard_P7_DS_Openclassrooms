# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:56:03 2022

@author: Pierre
"""

# Import useful package
from fastapi import FastAPI
import joblib
import pandas as pd

# Creation of the API
app = FastAPI()

# Root route
@app.get("/")
async def root():
    return{'message':'API Credit Scoring'}

# Import model and datas
model_dict = joblib.load('data/model_dict.joblib')
model = model_dict['model']
features = model_dict['features']
df = pd.read_csv('data/X_test.csv', index_col='SK_ID_CURR')
info_df = pd.read_csv('data/info.csv', index_col='SK_ID_CURR')

# Return information on one credit from its ID in pd.DataFrame format
def select_data(index):
    data = pd.DataFrame(df.loc[index, :]).T
    return data

# Return a score (from 1 to 4) from a predict_proba result
def get_score(proba):
    if proba <= 0.072:
        score = 1
    elif proba > 0.072 and proba <= 0.12:
        score = 2
    elif proba > 0.12 and proba <= 0.185:
        score = 3
    else:
        score = 4
    return score

# Calculation of probability to not repay credit from credit ID
# then returning corresponding score
@app.get("/scoring")
def scoring(Credit_ID: int):
    data = select_data(Credit_ID)
    prob = (model.predict_proba(data))[0][1]
    score = get_score(prob)
    result = {'probability': prob, 'score': score}
    return result