<div align="center">
  <img src="https://raw.githubusercontent.com/pierrepds/API_Dashboard_P7_DS_Openclassrooms/master/Dashboard/Images/Titre.png">
</div>
<br />
<br />

[Dashboard Scoring](https://dashboard-ocdsp7-pp.herokuapp.com) est un outil d'aide à la décision en matière de demande de crédit. 
Il est très simple et permet de visualiser:<br />
 - La probabilité de défaut d'un crédit sous la forme d'un score (de 1 à 4 - 4 étant la probabilité de défaut la plus forte).<br />
 - L'influence globale et locale de chaque variable dans la probabilité finale.<br />
 - La distribution de toutes les demandes de crédit pour chaque variable (analyse univariée et bivariée).<br />

## Modélisation

Le modèle de prédiction se base sur la base de données référence d'une compétition Kaggle 
([Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview))
dont le but était la prédiction de défauts sur un jeu test à partir d'informations variées.<br />
<br />
Pour une utilisation du notebook en local, les fichiers ".csv" disponible sur la page de la competition Kaggle doivent être 
préalablement télécharger dans un dossier "Input" qui se trouvera dans le même dossier que le notebook.<br />
<br />
Le prétraitement est issue du kernel kaggle: [LightGBM with Simple Features](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script)

## API

Le modèle entrainé, disponible dans le dossier "data" permet de mettre en place une fonction de calcul de la probabilité de défaut. 
Cette probabilité est ensuite convertie en score puis la probabilité et le score sont renvoyés par 
[l'API](https://ocdsp7-api-pp.herokuapp.com/docs#).<br />
Les autres fonctions de [l'API](https://ocdsp7-api-pp.herokuapp.com/docs#) permettent de renvoyer:<br />
 - Un nouveau score après avoir modifié les valeurs de l'annuité et du montant du crédit.<br />
 - Les variables les plus importante localement.<br />
 - La distribution de tous les scores de l'échantillon test.<br />

## Dashboard

[Dashboard Scoring](https://dashboard-ocdsp7-pp.herokuapp.com) est l'outil qui permet de visualiser les résultats de la modélisation 
via les résultats des fonctions de l'API. Il permet de visualiser d'un seul coup d'oeil le score de chaque demande de crédit. 
En plus, il permet de:<br />
 - Modifier les valeurs de l'annuité et du montant du crédit et de visualiser le nouveau score.<br />
 - Visualiser la distribution de l'échantillon test sur les variables les plus importantes globalement et localement et de visualiser 
la position de la demande traitée.<br />
 - Faire des analyses univariés et bivariés pour toutes les variables.<br />
