# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 01:14:51 2022

@author: Pierre
"""

# Import useful package
import requests
import json
import streamlit as st
import pandas as pd

# Import model and datas
df_test = pd.read_csv('data/X_test.csv', index_col='SK_ID_CURR')
info_df = pd.read_csv('data/info.csv', index_col='SK_ID_CURR')
infos = {'Montant du crédit':'AMT_CREDIT',
         'Montant des annuités':'AMT_ANNUITY',
         'Montant des biens concernés par le crédit':'AMT_GOODS_PRICE',
         'Type de crédit':'NAME_CONTRACT_TYPE',
         'Profession':'OCCUPATION_TYPE',
         'Revenus totaux':'AMT_INCOME_TOTAL',
         'Propriétaire d\'un logement':'FLAG_OWN_REALTY',
         'Sexe':'CODE_GENDER',
         'Situation familiale':'NAME_FAMILY_STATUS',
         'Dettes totales sur crédit en cours':'BURO_AMT_CREDIT_SUM_DEPT_SUM'
         }

# Get response from API
@st.cache
def scoring(Credit_ID: int):
    r = requests.get('https://rocky-tundra-36789.herokuapp.com/scoring',
                     {'Credit_ID':Credit_ID})
    response = r.json()
    return response

def custom_scoring(Credit_ID: int, new_ann, new_amt):
    r = requests.get('https://rocky-tundra-36789.herokuapp.com/custom_scoring',
                     {'Credit_ID':Credit_ID,
                      'new_ann':new_ann,
                      'new_amt':new_amt}
                     )
    response = r.json()
    return response


def score_dis():
    r = requests.get('https://rocky-tundra-36789.herokuapp.com/score_values_distribution')
    response = r.json()
    scr_df = pd.DataFrame(json.loads(response['scores']),
                          index=json.loads(response['index']),
                          columns=['score'])
    return scr_df

# Construction of the Dashboard
# Title

st.image('Images/Titre.png', use_column_width='always')

