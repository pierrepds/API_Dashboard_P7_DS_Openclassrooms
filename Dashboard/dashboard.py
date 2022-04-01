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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Import model and datas
df_test = pd.read_csv('data/X_test.csv', index_col='SK_ID_CURR')
info_df = pd.read_csv('data/info.csv', index_col='SK_ID_CURR')
gloss = pd.read_csv('data/gloss.csv', index_col='Unnamed: 0')
infos = {'Montant du crédit':'AMT_CREDIT',
         'Montant des annuités':'AMT_ANNUITY',
         'Montant des biens concernés par le crédit':'AMT_GOODS_PRICE',
         'Type de crédit':'NAME_CONTRACT_TYPE',
         'Profession':'OCCUPATION_TYPE',
         'Revenus totaux':'AMT_INCOME_TOTAL',
         'Propriétaire d\'un logement':'FLAG_OWN_REALTY',
         'Sexe':'CODE_GENDER',
         'Situation familiale':'NAME_FAMILY_STATUS',
         'Dettes totales sur crédit en cours':'BURO_AMT_CREDIT_SUM_DEBT_SUM'
         }

# Construction of functions
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

@st.cache
def feat_imp(Credit_ID: int):
    r = requests.get('https://ocdsp7-api-pp.herokuapp.com/local_feat_imp',
                     {'Credit_ID':Credit_ID})
    response = r.json()
    return response

# Distribution visualization
def dist_viz(data, id_credit, feature):
    scr_df = score_dis()
    df = data[feature]
    color = ['g', 'greenyellow', 'navajowhite', 'orange']

    # Distribution of missing values if value is missing
    if np.isnan(data.loc[id_credit, feature]):
        nb_nan = np.isnan(df).sum()
        nb_notnan = len(df) - nb_nan
        fig = plt.figure(figsize=(4,3))
        for scr, clr in zip([1, 2, 3, 4], color):
          sample = df.loc[scr_df.loc[df.index, 'score'] == scr]
          nan_nb = np.isnan(sample).sum() / nb_nan
          notnan_nb = (len(sample) - np.isnan(sample).sum()) / nb_notnan
          plt.bar([0.76 + ((scr-1) * 0.16), 1.76 + ((scr-1) * 0.16)],
                  [nan_nb, notnan_nb],
                  label=f'score = {scr}',
                  width=0.12,
                  color=clr
                  )
        plt.hlines(y=1, xmin=0.65, xmax=1.35, color='b', lw=2,
                    label=f'credit n°{id_credit}')
        plt.vlines(x=0.65, ymin=0, ymax=1, color='b', lw=2)
        plt.vlines(x=1.35, ymin=0, ymax=1, color='b', lw=2)
        plt.title('Distribution des scores\nValeurs manquantes',
                  fontsize=16, fontweight=650)
        plt.ylabel('Distribution', fontsize=14)
        plt.xticks([1,2], ['NaN', 'not NaN'])
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

    else:
        # Distribution visualization as bar if the feature is discrete
        if len(df.unique()) < 20:
            df.dropna(inplace=True)
            values = np.sort(df.unique())
            nb_values = len(values)
            value_client = df[id_credit]
            idx_client = np.where(values == value_client)[0][0]
            fig = plt.figure(figsize=(2*nb_values, 3))
            count = df.value_counts()
            for scr, clr in zip([1, 2, 3, 4], color):
                sample = df.loc[scr_df.loc[df.index, 'score'] == scr]
                val_cnt = sample.value_counts()
                y = []
                for val in values:
                    if val in val_cnt.index:
                        y.append(val_cnt[val]/count[val])
                    else:
                        y.append(0)
                plt.bar(np.arange(0.76 + ((scr-1) * 0.16),
                                  nb_values + 0.76 + ((scr-1) * 0.16),
                                  1),
                        y,
                        label=f'score = {scr}',
                        width=0.12,
                        color=clr
                        )
            plt.hlines(y=1,
                       xmin=0.65 + idx_client,
                       xmax=1.35 + idx_client,
                       color='b', lw=2,
                       label=f'credit n°{id_credit}')
            plt.vlines(x=0.65 + idx_client, ymin=0, ymax=1, color='b', lw=2)
            plt.vlines(x=1.35 + idx_client, ymin=0, ymax=1, color='b', lw=2)
            plt.title('Distribution des scores\nVariable discrète',
                      fontsize=16, fontweight=650)
            plt.ylabel('Distribution', fontsize=14)
            plt.xticks(range(1, 1+nb_values), values)
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

        # Distribution plot for continous features
        # Different ploting in fonction of values distribution
        else:
            df.dropna(inplace=True)
            nb_values = len(df.unique())
    
            # In case of small amount of unique values (almost discrete)
            if (nb_values <= 1000 
                and df.value_counts().iloc[0] < 1.5*len(df)/(nb_values)):
                out, bins = pd.qcut(df, nb_values/2,
                                    labels=False, retbins=True)
                X = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
                fig = plt.figure()
                for scr, clr in zip([1, 2, 3, 4], color):
                    y=[]
                    for i, count in zip(np.unique(out, return_counts=True)[0],
                                        np.unique(out, return_counts=True)[1]):
                        score_list = scr_df.loc[df[out == i].index]
                        ratio_score = score_list.value_counts()
                        if scr in ratio_score.index:
                            y.append(ratio_score[scr]/count)
                        else:
                            y.append(0)
                    plt.plot(X, y, c=clr, label=f'score = {scr}')
                plt.axvline(x=df[id_credit], ymin=0, ymax=1,
                           color='b', lw=2, label=f'credit n°{id_credit}')
                plt.title('Distribution des scores\nVariable continue',
                          fontsize=16, fontweight=650)
                plt.ylabel('Distribution', fontsize=14)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

            # In case of multiple unique values well distributed    
            elif (nb_values > 1000
                  and df.value_counts().iloc[0] < 1.5*len(df)/(nb_values/100)):
                out, bins = pd.qcut(df, int(nb_values / 100),
                                    labels=False, retbins=True)
                X = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
                fig = plt.figure()
                for scr, clr in zip([1, 2, 3, 4], color):
                    y=[]
                    for i, count in zip(np.unique(out, return_counts=True)[0],
                                        np.unique(out, return_counts=True)[1]):
                        score_list = scr_df.loc[df[out == i].index]
                        ratio_score = score_list.value_counts()
                        if scr in ratio_score.index:
                            y.append(ratio_score[scr]/count)
                        else:
                            y.append(0)
                    plt.plot(X, y, c=clr, label=f'score = {scr}')
                plt.axvline(x=df[id_credit], ymin=0, ymax=1,
                           color='b', lw=2, label=f'credit n°{id_credit}')
                plt.title('Distribution des scores\nVariable continue',
                          fontsize=16, fontweight=650)
                plt.ylabel('Distribution', fontsize=14)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

            # In case of imbalanced distribution    
            else:
                out, bins = pd.cut(df, 50, labels=False, retbins=True)
                X = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
                fig = plt.figure()
                for scr, clr in zip([1, 2, 3, 4], color):
                    y=[]
                    for i in range(len(X)):
                        if i in np.unique(out):
                            count = np.sum(list(out == i))
                            score_list = scr_df.loc[df[out == i].index]
                            ratio_score = score_list.value_counts()
                            if scr in ratio_score.index:
                                y.append(ratio_score[scr]/count)
                            else:
                                y.append(0)
                        else:
                            y.append(0)
                    plt.plot(X, y, c=clr, label=f'score = {scr}')
                plt.axvline(x=df[id_credit], ymin=0, ymax=1,
                           color='b', lw=2, label=f'credit n°{id_credit}')
                plt.title('Distribution des scores\nVariable continue',
                          fontsize=16, fontweight=650)
                plt.ylabel('Distribution', fontsize=14)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    return fig

# Bivariate analysis visualization
def bin_viz(data, id_credit, abscissa, ordinate):
    # Definition of legend
    scr1 = mlines.Line2D([], [], color=(0, 0, 0, 0), mec='g',
                         marker='o', linestyle='None',
                         markersize=5, label='score = 1', mew=2)
    scr2 = mlines.Line2D([], [], color=(0, 0, 0, 0), mec='greenyellow',
                         marker='o', linestyle='None',
                         markersize=5, label='score = 2', mew=2)
    scr3 = mlines.Line2D([], [], color=(0, 0, 0, 0), mec='navajowhite',
                         marker='o', linestyle='None',
                         markersize=5, label='score = 3', mew=2)
    scr4 = mlines.Line2D([], [], color=(0, 0, 0, 0), mec='orange',
                         marker='o', linestyle='None',
                         markersize=5, label='score = 4', mew=2)
    size_1 = mlines.Line2D([], [], color=(0, 0, 0, 0), mec='black',
                         marker='o', linestyle='None', mew=2,
                         markersize=5, label='nombre d\'observation = 1')
    size_5 = mlines.Line2D([], [], color=(0, 0, 0, 0), mec='black',
                         marker='o', linestyle='None', mew=2,
                         markersize=10, label='nombre d\'observation = 5')
    size_10 = mlines.Line2D([], [], color=(0, 0, 0, 0), mec='black',
                         marker='o', linestyle='None', mew=2,
                         markersize=15, label='nombre d\'observation = 10')
    point = mlines.Line2D([], [], color=(0, 0, 0, 0), c='b', marker='+',
                          linestyle='None',markersize=10,
                          label='credit n°{id_credit}')

    color = ['g', 'greenyellow', 'navajowhite', 'orange']
    df = data[[abscissa, ordinate]].copy()
    df['score'] = score_dis()['score']
    df.dropna(inplace=True)
    color = ['g', 'greenyellow', 'navajowhite', 'orange']

    # Scatter plot with size in fonction of number of values
    # for discrete features
    if len(df[abscissa].unique()) < 10 and len(df[ordinate].unique()) < 10:
        sizes = {}
        for x, y, scr in zip(df[abscissa], df[ordinate], df['score']):
          if (x, y, scr) in sizes:
            sizes[(x, y, scr)] += 1
          else:
            sizes[(x, y, scr)] = 5
        keys = sizes.keys()
        fig = plt.figure()
        plt.scatter(x=[k[0] for k in keys],
                    y=[k[1] for k in keys],
                    s=[sizes[k] for k in keys],
                    c=[(0, 0, 0, 0) for _ in keys],
                    ec=[color[k[2]-1] for k in keys],
                    lw=2)
        plt.title('Analyse bivariée\nVariables discrètes',
                fontsize=16, fontweight=650)

    # Boxplot when mixing a discrete and a continuous features
    elif len(df[abscissa].unique()) < 10 and len(df[ordinate].unique()) >= 10:
        fig = plt.figure()
        sns.boxplot(data=df, x=abscissa, y=ordinate,
                    orient='v', hue='score', palette=color)
        plt.title('Analyse bivariée\nVariables discrète et continue',
                  fontsize=16, fontweight=650)

    elif len(df[abscissa].unique()) >= 10 and len(df[ordinate].unique()) < 10:
        fig = plt.figure()
        sns.boxplot(data=df, x=abscissa, y=ordinate,
                    orient='h', hue='score', palette=color)
        plt.title('Analyse bivariée\nVariables discrète et continue',
                  fontsize=16, fontweight=650)

    # Scatter plot for two continuous features
    else:
      sizes = {}
      for x, y, scr in zip(df[abscissa], df[ordinate], df['score']):
        if (x, y, scr) in sizes:
          sizes[(x, y, scr)] += 1
        else:
          sizes[(x, y, scr)] = 5
      keys = sizes.keys()
      fig = plt.figure()
      plt.scatter(x=[k[0] for k in keys],
                  y=[k[1] for k in keys],
                  s=[sizes[k] for k in keys],
                  c=[(0, 0, 0, 0) for _ in keys],
                  ec=[color[k[2]-1] for k in keys],
                  lw=2)
      plt.title('Analyse bivariée\nVariables continues',
                fontsize=16, fontweight=650)

    if id_credit in df.index:
        plt.axvline(x=df.loc[id_credit, abscissa], ymin=0, ymax=1, color='b')
        plt.axhline(y=df.loc[id_credit, ordinate], xmin=0, xmax=1, color='b')
    plt.ylabel(ordinate, fontsize=14)
    plt.xlabel(abscissa, fontsize=14)
    plt.legend(handles=[scr1, scr2, scr3, scr4,
                        size_1, size_5, size_10,
                        point],
               bbox_to_anchor=(1, 1), loc='upper left')
    
    return fig


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

st.sidebar.markdown('#')
feat_gloss = st.sidebar.selectbox('Glossaire des variables', gloss.index)
st.sidebar.markdown(gloss.loc[feat_gloss, 'description'])

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

# Visualize global features importance and their sample distribution
if analyse == 'Analyse global':
    st.header('Analyse global du modèle: ')
    st.image('Images/shap_values.png', use_column_width='always')
    st.image('Images/shap_values_relative.png', use_column_width='always')
    st.header('Situation crédit')
    important_features = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1',
                          'NAME_FAMILY_STATUS_Married', 'CODE_GENDER',
                          'PAYMENT_RATE', 'INSTAL_AMT_PAYMENT_SUM',
                          'INSTAL_DPD_MEAN',
                          'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 'AMT_ANNUITY'
                          ]
    feature = st.selectbox('Sélection de la variable à visualiser:',
                           important_features)

    fig = dist_viz(data=df_test, id_credit=id_client, feature=feature)
    st.pyplot(fig)
    
# Visualize sample distribution for the most important local features
if analyse == 'Analyse locale':
    st.header('Analyse locale du modèle: ')
    local_imp = feat_imp(id_client)
    st.subheader('Principales raisons de refus:')
    col1, col2 = st.columns(2)
    col1.markdown('Cause du refus n°1: {}'
                 ''.format(local_imp['feat_pos_1']))
    col1.markdown('Proportion de la variable dans la décision: {}%'
                 ''.format(round(local_imp['ratio_pos_1']*100, 1)))
    fig = dist_viz(data=df_test,
                   id_credit=id_client,
                   feature=local_imp['feat_pos_1'])
    col1.pyplot(fig)
    col2.markdown('Cause du refus n°2: {}'
                 ''.format(local_imp['feat_pos_2']))
    col2.markdown('Proportion de la variable dans la décision: {}%'
                 ''.format(round(local_imp['ratio_pos_2']*100, 1)))
    fig = dist_viz(data=df_test,
                   id_credit=id_client,
                   feature=local_imp['feat_pos_2'])
    col2.pyplot(fig)
    
    st.subheader('Principales raisons d\'accord:')
    col1, col2 = st.columns(2)
    col1.markdown('Cause d\'accord n°1: {}'
                 ''.format(local_imp['feat_neg_1']))
    col1.markdown('Proportion de la variable dans la décision: {}%'
                 ''.format(round(local_imp['ratio_neg_1']*100, 1)))
    fig = dist_viz(data=df_test,
                   id_credit=id_client,
                   feature=local_imp['feat_neg_1'])
    col1.pyplot(fig)
    col2.markdown('Cause d\'accord n°2: {}'
                 ''.format(local_imp['feat_neg_2']))
    col2.markdown('Proportion de la variable dans la décision: {}%'
                 ''.format(round(local_imp['ratio_neg_2']*100, 1)))
    fig = dist_viz(data=df_test,
                   id_credit=id_client,
                   feature=local_imp['feat_neg_2'])
    col2.pyplot(fig)

# Visualize sample distribution in a scatter plot allowing bivariate analysis
if analyse == 'Analyse bivariée':
    st.header('Analyse bivariée:')
    col1, col2 = st.columns(2)
    absc = col1.selectbox('Abscisse',
                          df_test.columns)
    ordo = col2.selectbox('Ordonnée',
                          df_test.columns)
    
    if absc == ordo:
        fig = dist_viz(data=df_test, id_credit=id_client, feature=absc)
        st.pyplot(fig)
    else:
        fig = bin_viz(data=df_test, id_credit=id_client,
                      abscissa=absc, ordinate=ordo)
        st.pyplot(fig)