# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:56:03 2022

@author: Pierre
"""

# Import useful package
from fastapi import FastAPI
import joblib
import shap
import json
import pandas as pd
import numpy as np

# Creation of the API
app = FastAPI()

# Root route
@app.get("/")
async def root():
    return{'message':'API Credit Scoring'}

# Import model and datas
model_dict = joblib.load('../data/model_dict.joblib')
model = model_dict['model']
features = model_dict['features']
df = pd.read_csv('../data/X_test.csv', index_col='SK_ID_CURR')
info_df = pd.read_csv('../data/info.csv', index_col='SK_ID_CURR')

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

# New calculation of probability and score by changing some features value
@app.get("/custom_scoring")
def custom_scoring(Credit_ID: int, new_ann: float, new_amt: float):
    data = select_data(Credit_ID)
    data.loc[Credit_ID, 'AMT_ANNUITY'] = new_ann
    amt_income = info_df.loc[Credit_ID, 'AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_PERC'] = new_ann / amt_income
    data['PAYMENT_RATE'] = new_ann / new_amt
    prob = (model.predict_proba(data))[0][1]
    score = get_score(prob)
    result = {'probability': prob, 'score': score}
    return result

# Calculation of feature importance and extraction of the most important ones
# (with negative and positive effect)
@app.get("/local_feat_imp")
def feat_imp(Credit_ID: int):
    expl = shap.TreeExplainer(model[3])
    imp = model[0]
    df_trans = pd.DataFrame(imp.transform(df),
                               index=df.index,
                               columns=features)
    local_sv = expl.shap_values(pd.DataFrame(df_trans.loc[Credit_ID]).T)[1][0]
    sort_idx = np.argsort(local_sv)
    neg_val = np.sum([sv for sv in local_sv if sv<0])
    pos_val = np.sum([sv for sv in local_sv if sv>0])
    feat_neg_1 = features[sort_idx[0]]
    ratio_neg_1 = local_sv[sort_idx[0]] / neg_val
    feat_neg_2 = features[sort_idx[1]]
    ratio_neg_2 = local_sv[sort_idx[1]] / neg_val
    feat_pos_1 = features[sort_idx[-1]]
    ratio_pos_1 = local_sv[sort_idx[-1]] / pos_val
    feat_pos_2 = features[sort_idx[-2]]
    ratio_pos_2 = local_sv[sort_idx[-2]] / pos_val
    result = {'feat_neg_1':feat_neg_1, 'ratio_neg_1':ratio_neg_1,
              'feat_neg_2':feat_neg_2, 'ratio_neg_2':ratio_neg_2,
              'feat_pos_1':feat_pos_1, 'ratio_pos_1':ratio_pos_1,
              'feat_pos_2':feat_pos_2, 'ratio_pos_2':ratio_pos_2}
    return result

# Return the list of scores for the test sample
@app.get("/score_values_distribution")
def scr_dist():
    scr_idx = list(df.index)
    proba_ls = list(model.predict_proba(df)[:,1])
    scr_ls = [get_score(proba) for proba in proba_ls]
    result = {'index':json.dumps(scr_idx),
              'scores':json.dumps(scr_ls)}
    return result
    