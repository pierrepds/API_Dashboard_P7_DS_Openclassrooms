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
    r = requests.get('https://ocdsp7-api-pp.herokuapp.com/scoring',
                     {'Credit_ID':Credit_ID})
    response = r.json()
    return response

def custom_scoring(Credit_ID: int, new_ann, new_amt):
    r = requests.get('https://ocdsp7-api-pp.herokuapp.com/custom_scoring',
                     {'Credit_ID':Credit_ID,
                      'new_ann':new_ann,
                      'new_amt':new_amt}
                     )
    response = r.json()
    return response

@st.cache
def score_dis():
    r = requests.get('https://ocdsp7-api-pp.herokuapp.com/'
                     'score_values_distribution')
    response = r.json()
    scr_df = pd.DataFrame(json.loads(response['scores']),
                          index=json.loads(response['index']),
                          columns=['score'])
    return scr_df

# Construction of the Dashboard
# Title
st.image('Images/Titre.png', use_column_width='always')

# Sidebar
id_client = st.sidebar.selectbox('Sélection de l\'identifiant du crédit:',
                                 df_test.index)
st.sidebar.markdown('#')

info_col = st.sidebar.multiselect('Information du client à afficher',
                                  infos.keys(),
                                  default=['Type de crédit',
                                           'Montant du crédit',
                                           'Montant des annuités']
                                  )

st.sidebar.table(info_df.astype(str).loc[id_client,
                                         [infos[key] for key in info_col]])

analyse = st.sidebar.radio('Analyse',
                           ['Modification valeurs', 'Analyse global',
                            'Analyse locale', 'Analyse bivariée'])

# Visualisation of scoring results
st.header('Résultat de la simulation')
st.markdown('##')
score = scoring(id_client)['score']
st.image(f'Images/ScoreIm{score}.png', use_column_width='always')

# Custom results by changing annuity and credit amount
if analyse == 'Modification valeurs':
    st.header('Modification Annuités / Montant du crédit')
    col1, col2 = st.columns(2)
    new_annuity = col1.slider('Changer la valeur des annuités',
                              max_value=int(2*df_test.loc[id_client,
                                                          'AMT_ANNUITY']),
                              value=int(df_test.loc[id_client, 'AMT_ANNUITY']))
    new_amount = col1.slider('Changer la valeur du montant du crédit',
                              max_value=int(2*info_df.loc[id_client,
                                                          'AMT_CREDIT']),
                              value=int(info_df.loc[id_client, 'AMT_CREDIT']))
    custom_score = custom_scoring(id_client, new_annuity, new_amount)['score']
    col2.markdown('##')
    col2.image(f'Images/Score{custom_score}.png')