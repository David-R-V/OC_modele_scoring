from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse,HTMLResponse
from fastapi.responses import Response
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from lightgbm import LGBMClassifier
import gzip
from itertools import islice
import shap
import matplotlib.pyplot as plt
import lightgbm
import os


app = FastAPI()

port=int(os.environ.get('PORT', 8000))

serie_index_clients = pd.read_csv('2_backend/API_input_data/Liste_index_clients_futur.csv',index_col='ID')

#model_risk = joblib.load('2_backend/API_input_data/model_risk_100.joblib')

# def d'un endpoint "id_client" qui possède la liste de tous les id dans un fichier sans devoir lire la database totale
@app.get("/id_clients")
async def get_id_clients():
    return FileResponse("2_backend/API_input_data/Liste_index_clients_futur.csv")

# def d'un endpoint "data"=ptf filtré sur colonnes sélectionnées pour graphiques comparaisons entre ID et ptf
@app.get("/data")
async def get_df_infos():
    # Sélection des colonnes pour lesquelles on affichera la comparaison ID/ptf + 'ID' (ID doit etre inclus dans la liste à charger avant d'etre déclaré comme index)
    col_infos_globales=['ID','AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE','CODE_GENDER', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT']
    # Chargement des infos globales sur certaines variables pour comparaison ID/ptf
    df_infos=pd.read_csv("2_backend/API_input_data/base_futur.csv",sep='\t',compression='gzip',index_col='ID',usecols=col_infos_globales)
    csv_infos = df_infos.to_csv(index=True)
    return Response(content=csv_infos, media_type="text/csv")

# def d'un endpoint "informations sur l'ID indiqué sous la forme d'un dataframe"

info_base=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
            'AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED',
            'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']

@app.get("/data/{index_cible}")
async def get_pf_data(index_cible: int):
     
    file_path = "2_backend/API_input_data/base_futur.csv"
    separator = '\t'
    compression = 'gzip'

    try:
        line_number = serie_index_clients.index.get_loc(index_cible) + 1
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Index {index_cible} not found")

    with gzip.open(file_path, mode='rt', encoding='latin1') as file:
        selected_line = next(islice(file, line_number, line_number + 1), None)

    if selected_line is not None:
        if selected_line.endswith('\n') and not selected_line.strip().endswith(','):
            selected_line = selected_line.rstrip('\n')

        selected_line = pd.DataFrame(selected_line.split(separator),
                                     index=pd.read_csv(file_path, sep=separator, compression=compression, nrows=1).columns.to_list()).T.set_index('ID')
        selected_line = selected_line.replace('', np.nan)
        selected_line = selected_line[info_base].round(3)

        csv_ligne_select = selected_line.to_csv(index=True)
        return Response(content=csv_ligne_select, media_type="text/csv")
    else:
        return Response(content={"message": "Erreur: index inconnu"})

@app.get("/get_html_content", response_class=HTMLResponse)
async def get_html_content():
    try:
        with open('2_backend/API_input_data/report_5.html', encoding='utf-8') as report_f:
            report_text = report_f.read()
            return HTMLResponse(content=report_text, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML file not found")

Endpoint qui renvoie la proba associée à l'ID sélectionnée
@app.get("/model_predictif/{index_cible}")

async def model_predictif(index_cible: int) -> List[float] :
         
    file_path = "2_backend/API_input_data/base_futur_prep_algo.csv"
    separator = '\t'
    compression = 'gzip'

    try:
        line_number = serie_index_clients.index.get_loc(index_cible) + 1
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Index {index_cible} not found")

    with gzip.open(file_path, mode='rt', encoding='latin1') as file:
        selected_line = next(islice(file, line_number, line_number + 1), None)

    if selected_line is not None:
        if selected_line.endswith('\n') and not selected_line.strip().endswith(','):
            selected_line = selected_line.rstrip('\n')

        selected_line = pd.DataFrame(selected_line.split(separator),
                                     index=pd.read_csv(file_path, sep=separator, compression=compression, nrows=1).columns.to_list()).T.set_index('ID')
        selected_line = selected_line.replace('', np.nan)
        selected_line = selected_line.drop(['TARGET','INDEX','SK_ID_CURR'],axis=1)
        selected_line = selected_line.round(3)

        # on a techniquement besoin que de la proba de classe 1 : si >0.5, classe 1 = pb de paiement; si <0.5, classe 0
        proba_risk = model_risk.predict_proba(selected_line.values.reshape(1, -1))[:,1]

        return proba_risk
    
    else:
        return Response(content={"message": "Erreur: index inconnu"})

#Import de l'explication SHAP
try:
    explanation = joblib.load("2_backend/API_input_data/shap_explanation_model_risk_100.joblib")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading explanation file: {str(e)}")

#Endpoint qui génère l'image Waterfall plot et renvoie le texte du chemin où elle se trouve
@app.get("/explain/{index_cible}")
async def explain(index_cible: int):
    try:
        line_number = serie_index_clients.index.get_loc(index_cible) + 1
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Index {index_cible} not found")

    try:
        shap_values = explanation[line_number]
        
        shap.waterfall_plot(shap_values,max_display=20,show=False)
        title=plt.title('',loc='left')
        title.set_position((-0.4,0))
        #title_text_obj = plt.text(-0.13,0.88, f"ID = {index_cible}", color='blue', fontsize=12, fontweight='bold', transform=fig.transFigure)
        plt.savefig('2_backend/API_input_data/shap_waterfall_plot.jpeg',bbox_inches='tight')
        plt.close()
        
        return "2_backend/API_input_data/shap_waterfall_plot.jpeg"
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   

Endpoint qui génère l'image des features rangées par importance lgbm et renvoie le texte du chemin où elle se trouve
@app.get("/explain")
async def explain():

    try: 
        lightgbm.plot_importance(model_risk,max_num_features=30,importance_type='split',figsize=(8,8),title='Classement des variables impactantes globales du modèle')
        plt.savefig('2_backend/API_input_data/lgbm_features_plot.jpeg',bbox_inches='tight')
        plt.close()
        
        return "2_backend/API_input_data/lgbm_features_plot.jpeg"
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 


    
















