**Répertoire de travail du projet 7 : Implémentation d'un modèle de scoring**

L'objectif de ce projet est le déploiement en ligne d'un dashboard utilisant un modèle de machine learning prédictif de credit score.
Le dashboard doit être conçu dans l'optique d'une utilisation côté organisme bancaire afin de présenter des informations à l'utilisateur en charge du dossier.
Il doit contenir:
- des informations relatives au client demandeur du prêt,
- le résultat prédictif du modèle quant au risque du prêt,
- des visualisations relatives à l'interprétabilité de la décision du modèle,
- le rapport du data drift potentiel entre la base d'entrainement et le base actuelle (data drift via Evidently)

Rappel des liens:
Répertoire github
https://github.com/David-R-V/OC_modele_scoring.git

Dashboard en ligne:
https://drv-p7-front-cb9d219d05d3.herokuapp.com/

La première partie du projet consiste à choisir et entrainer le modèle sur la base des données fournies

En deuxième partie, on a ensuite défini les scripts relatifs à l'API (FastAPI utilisée) et au dashboard (Streamlit) pour un déploiement en ligne via Heroku.

Sur Heroku, le fonctionnement est séparé en 2 applications:
- 'drv-p7-backend' contient l'API et les fichiers sources. Elle effectue les calculs et prend les informations à communiquer au dashboard
- 'drv-p7-front' contient le dashboard, qui effectue des requêtes auprès de l'API et affiche les informations voulues

**Contenu du répertoire:**

'Projet 7.ipynb' est le script python au format jupyte notebook ayant permis l'exploration des données, la recherche et l'entrainement du modèle, la génération des différentes fichiers sources de l'API et du rapport de data drift

1_outputs/ est un répertoire de stockage des différents outputs générés au fil du développement par 'Projet 7.ipynb'

2_backend/ est le répertoire du script de l'API et des fichiers source qu'elle utilise
	- API.py est le script python de l'API
	- test_fichiers_app.py est le script python des test Pytest qui sont effectués via GitHub Action à chaque push
	- API_input_data/ est le répertoire des fichiers source utilisés par l'API

3_frontend/ est le répertoire du script du dashboard
	- Dashboard.py est le script pyhton du dashboard streamlit
	- Procfile,requirements.txt et runtime.txt sont des fichiers de paramétrage pour Heroku de l'application 'drv-p7-front'


Procfile, requirements.txt et runtime.txt sont des fichiers de paramétrage pour Heroku de l'application 'drv-p7-backend'