from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import Response
import pandas as pd
import numpy as np
import gzip
from itertools import islice

app = FastAPI()

# def d'un endpoint "data=ptf total sur colonnes sélectionnées pour graphiques ID vs ptf"
@app.get("/id_clients")
async def get_id_clients():
    return FileResponse("../1_outputs/Liste_index_clients_futur.csv")





@app.get("/data")
async def get_df_infos():
    # Sélection des colonnes pour lesquelles on affichera la comparaison ID/ptf + 'ID' (ID doit etre inclus dans la liste à charger avant d'etre déclaré comme index)
    col_infos_globales=['ID','AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE','CODE_GENDER', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT']
    # Chargement des infos globales sur certaines variables pour comparaison ID/ptf
    df_infos=pd.read_csv("../1_outputs/base_futur.csv",sep='\t',compression='gzip',index_col='ID',usecols=col_infos_globales)
    csv_infos = df_infos.to_csv(index=True)
    return Response(content=csv_infos, media_type="text/csv")

# def d'un endpoint "informations sur l'ID indiqué sous la forme d'un dataframe"

info_base=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
            'AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED',
            'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']


@app.get("/data/{index_cible}")
async def get_pf_data(index_cible: int):
     
    file_path = "../1_outputs/base_futur.csv"
    separator = '\t'
    compression = 'gzip'

    serie_index_clients = pd.read_csv("../1_outputs/Liste_index_clients_futur.csv",index_col='ID')

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
