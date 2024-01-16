# Imports de librairies
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import ast
import seaborn as sns
from PIL import Image
from scipy import stats
import requests
from io import StringIO
import streamlit.components.v1 as components
import os

###########################################################################
###    Définitions diverses: api, variables, fonctions, endpoints....   ###
###########################################################################

# Mise en page large de base
# permet de mieux controler l'affichage avec colonnes vides au besoin.
port = int(os.environ.get("PORT", 8000))

st.set_page_config(layout="wide")

info_base=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
            'AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED',
            'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']

#serie_index_clients = pd.read_csv("../2_backend/API_input_data/Liste_index_clients_futur.csv",index_col='ID')

# Adresse de l'API
api_base_url = "https://drv-p7-backend-4265cdd1ff0b.herokuapp.com" 

# Endpoint df_infos
endpoint_df_infos = f"{api_base_url}/data"
response_infos = requests.get(endpoint_df_infos)
if response_infos.status_code == 200:
    df_infos=pd.read_csv(StringIO(response_infos.text),index_col='ID')
else:
    st.error(f"Erreur d'accès à endpoint df_infos, code {response_infos.status_code}")

# Endpoint rapport du data_drift
endpoint_data_drift = f"{api_base_url}/get_html_content"
response_data_drift = requests.get(endpoint_data_drift)

# Endpoint données input prédiction
endpoint_data_pred = f"{api_base_url}/model_predictif"

# Fonction ecretage outliers 10x écarts interquartile(25-75)
def clip_quart(col):
    return col.clip(upper=np.percentile(col.loc[col.isna()==False],75)+
                    10*(np.percentile(col.loc[col.isna()==False],75)-np.percentile(col.loc[col.isna()==False],25)),
                    lower=np.percentile(col.loc[col.isna()==False],25)-
                    10*(np.percentile(col.loc[col.isna()==False],75)-np.percentile(col.loc[col.isna()==False],25)))

# Fonction graphique sur distributions contines: densité + affichage point index client
def kde_clip(index_to_highlight,col,nom_fichier):

    plt.figure(figsize=(4,4))
    ax=sns.kdeplot(clip_quart(df_infos[col]),bw_adjust=2,fill=True)
    plt.xlim(df_infos[col].min(),np.percentile(df_infos[col],99.5))
    # Mise en avant de l'index indiqué
    if index_to_highlight is not None:
        highlight_value = df_infos.at[index_to_highlight, col]
        x_kde, y_kde = sns.kdeplot(clip_quart(df_infos[col]), bw_adjust=2).get_lines()[0].get_data()
        y_value = np.interp(highlight_value, x_kde, y_kde)
        # Placement du point
        plt.scatter(highlight_value, y_value, color='red', marker='o', s=50)  

        # Plot a vertical line from the KDE curve to the y-axis
        plt.plot([highlight_value, highlight_value], [0, y_value], color='red', linestyle='--')  # Vertical line

    # Paramètres   r"$\bf{" + str(number) + "}$"
    plt.title(f'{col} \n ID {index_to_highlight} = {highlight_value} \n {int(round(stats.percentileofscore(df_infos[col], highlight_value),0))}ème centile',
              fontsize=9)
    plt.xlabel(f'{col}',fontsize=8)
    plt.ylabel('Densité de répartition')
    ax.tick_params(labelleft=False)
    #plt.xticks(rotation=60)
    plt.tick_params(axis='x', labelsize=8)
    plt.gca().xaxis.get_offset_text().set_fontsize(8)

    plt.grid()
    plt.savefig(f'{nom_fichier}.png')

# Fonction qui crée le fichier graphe de densité via kde_clip puis renvoie le fichier ouvert
def img_kdeplot(selected_value,col,nom_fichier):

    kde_clip(selected_value,col,nom_fichier)
    image = Image.open(f'{nom_fichier}.png')

    return image

# Stockage d'une sélection aléatoire d'index clients à requeter
if 'random_sel' not in st.session_state:
    #accès à l'endpoint id_clients
    endpoint_id_clients = f"{api_base_url}/id_clients"
    response_id=requests.get(endpoint_id_clients)
    if response_id.status_code == 200:
            #créer l'objet serie_index_clients issus de la requete
            serie_index_clients=pd.read_csv(StringIO(response_id.text),index_col='ID')
    else:
        st.error(f"Erreur d'accès à endpoint id_clients, code {response_id.status_code}")

    st.session_state.random_sel = random.sample(list(serie_index_clients.index), 15)

# Sélection des colonnes pour lesqelles on affichera la comparaison ID/ptf + 'ID' (ID doit etre inclus dans la liste à charger avant d'etre déclaré comme index)
col_infos_globales=['ID','AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE','CODE_GENDER', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT']

# Fonction de graphique pie plot sur distribution catégories avec indication valeur index client
def pieplot(index_client, col,nom_fichier):
    # Variables de travail
    source = df_infos[col]
    effectifs = source.value_counts().sort_index()
    pourcents = [valeur / effectifs.sum() * 100 for valeur in effectifs]
    labels = effectifs.index
    highlighted_index = labels.get_loc(df_infos.at[index_client, col])  # index à mettre en avant dans le explode

    # Création du plot
    fig, ax = plt.subplots(figsize=(3, 3))
    wedges, texts, autotexts = ax.pie(effectifs, labels=labels,
                                      explode=[0 if i != highlighted_index else 0.1 for i in range(len(effectifs))],
                                      autopct='',
                                      labeldistance=0.5, textprops={'fontsize': 12}
                                      )

    # Mettre une part en surbrillance, bordure et ombre
    highlighted_wedge = wedges[highlighted_index]
    highlighted_wedge.set_edgecolor('black')
    shadow = patches.Shadow(highlighted_wedge, -0.03, -0.03, linewidth=1, alpha=0.7)
    ax.add_patch(shadow)

    # Legend
    legend_labels = [f'{label} ({round(pourcent, 1)}%)' for label, pourcent in zip(labels, pourcents)]
    ax.legend(wedges, legend_labels, loc='upper left',fontsize=8)

    # Titre
    ax.set_title(f'{col} - ID {index_client} = {df_infos.at[index_client, col]} ',fontsize=9,x=0.47)
    fig.savefig(f'{nom_fichier}.png')

def img_pieplot(selected_value,col,nom_fichier):

    pieplot(selected_value,col,nom_fichier)
    image = Image.open(f'{nom_fichier}.png')

    return image

##################################################
###    Mise en page et création du dashboard   ###
##################################################



# Paramètres de style du titre
# (pas de centrage comme option de base dans st.title())
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 45px;
        font-weight: bold
    }
    </style>
    """, unsafe_allow_html=True)

# Titre avec paramètres ci-dessus
st.markdown("<h1 class='centered-title'>Application de prédiction de risque de défaut individuel</h1>", unsafe_allow_html=True)

# Ecrire les éléments sous forme "col1,col2,..." pour gérer la position dans l'espace via des colonnes vides qui générent de l'espace dans le dashboard

## Boite du multiselect de l'index client ##
col1,col2,col3=st.columns([0.5,1,1])
with col2:
    selected_values = st.multiselect("Sélection de l'index ID:", st.session_state.random_sel)

## Limitation du nombre de sélections simultanées à 3 ##
max_selections = 3
if len(selected_values) > max_selections:
    st.warning(f"Maximum de {max_selections} sélections affichées atteint.")
    selected_values = selected_values[:max_selections] 

# Création des données affichées lorsque sélectionnées dans le multiselect 1
df_general=pd.DataFrame(columns=info_base,index=[0])
df_general=df_general.fillna(0)
if selected_values:
    for selected_value in selected_values:
        
        # accès aux données "ligne relative à l'ID sélectionné"
        endpoint_donnees_ID = f"{api_base_url}/data/{selected_value}"
        response_id_data = requests.get(endpoint_donnees_ID)

        # Vérif résussite de la requete
        if response_id_data.status_code == 200:
            ligne_infos=pd.read_csv(StringIO(response_id_data.text),index_col='ID')
            # ajout des infos de la requete au df_general
            df_general =pd.concat([df_general, ligne_infos])
        else:
            st.error(f"Erreur d'accès à endpoint id_data, code {response_id_data.status_code}")    
        
        
    df_general.drop([0],axis=0,inplace=True)
    df_general=df_general.T

## Section information générales ##
    col1,col2,col3=st.columns([0.5,1.5,0.5])
    with col2:  
        st.header("Informations générales",divider="orange")
        st.dataframe(df_general,width=600)
    
else:
    col1,col2,col3=st.columns([0.5,1,1])
    with col2:
        st.warning("No values selected.")

if selected_values:
    col1,col2,col3=st.columns([0.5,1.5,0.5])
    with col2:
        st.header("Positionnement de l'ID sélectionné - Catégories",divider="orange")
    for selected_value in selected_values:
        col1,col2,col3,col4,col5,col6,col7 = st.columns([0.5,0.3,0.3,0.3,0.3,0.3,0.5])
        for col_x,col_y in zip(['CODE_GENDER', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT'],[col2,col4,col6]):
            with col_y:            
                st.image(img_pieplot(selected_value,col_x,col_x),use_column_width=False)    

else:
    col1,col2,col3=st.columns([0.5,1.5,0.5])
    with col2:    
        st.warning("No values selected.")

if selected_values:
    col1,col2,col3=st.columns([0.5,1.5,0.5])
    with col2:
        st.header("Positionnement de l'ID sélectionné - Valeurs numériques",divider="orange")
    for selected_value in selected_values:
        col1,col2,col3,col4,col5,col6 = st.columns([0.5,1.5/4,1.5/4,1.5/4,1.5/4,0.5])
        for col_x,col_y in zip(['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE'],[col2,col4]):
            with col_y:            
                st.image(img_kdeplot(selected_value,col_x,col_x),use_column_width=False)

if selected_values:
    col1,col2,col3=st.columns([0.5,1.5,0.5])
    with col2:
        st.header("Résultats du modèle prédictif",divider="orange")

    # création d'un df de stockage
    #df_pred=pd.DataFrame(columns=['Proba_1','Classe'])

    # opérations pour chacunes des valeurs sélectionnées
    for selected_value in selected_values:
        
        # accès à la proba issue de la prédiction
        endpoint_donnees_ID_algo = f"{api_base_url}/model_predictif/{selected_value}"
        response_id_data_algo = requests.get(endpoint_donnees_ID_algo)

        # Vérif résussite de la requete
        if response_id_data_algo.status_code == 200:
            # mise en forme pour insert dans le df de stockage
            ligne_pred=pd.DataFrame([ast.literal_eval(response_id_data_algo.text)],columns=['Proba_1',],index=[selected_value])
            ligne_pred['Classe']=np.where(ligne_pred['Proba_1']>=0.5,1,0)
        else:
            st.error(f"Erreur d'accès à endpoint id_data_algo, code {response_id_data_algo.status_code}")    
        # ajout des infos de la requete au df_pred
        #df_pred=pd.concat([df_pred,ligne_pred])

        with col2:
            
            temp_1=np.where(ligne_pred.loc[selected_value,'Classe']==1,'Oui','Non')
            # transfo de la valeur classe en texte oui/non
            temp_2=np.where(ligne_pred.loc[selected_value,'Classe']==1,ligne_pred.loc[selected_value,'Proba_1'],1-ligne_pred.loc[selected_value,'Proba_1'])
            temp_2=np.round(temp_2*100,1)
            # transfo de la proba: si classe 1, on utilise proba_1, si classe 0, on utilise proba_0 = 1-proba_1 (on a importé que la proba d'appartenir à la classe 1 via l'api, la proba d'appartenir à la classe 0 est obtenue par différence du choix binaire)
            if temp_1=='Non':
                st.subheader(f":grey[ID Client:] :blue[{str(selected_value)}]")
            else:
                st.subheader(f":grey[ID Client:] :red[{str(selected_value)}]")
            

            st.subheader(":grey[Difficultés de paiement anticipées par le modèle:]")

            if temp_1=='Non':
                st.header(f":blue[{temp_1}]")
            else:
                st.header(f":red[{temp_1}]")

            st.subheader(":grey[Probabilité de confiance de l'algorithme:]")

            if temp_1=='Non':
                st.header(f":blue[{temp_2}%]")
            else:
                st.header(f":red[{temp_2}%]")

            if (len(selected_values)>1) and (selected_value!=selected_values[-1]):
                col4,col5,col6=st.columns([0.5,1.5,0.5])
                with col5:
                    st.header('',divider='gray')

if selected_values:

    col1,col2,col3=st.columns([0.5,1.5,0.5])
    with col2:
        st.header("Variables influentes dans la prévision de l'algorithme",divider="orange")
        col4,col5,col6=st.columns([0.5,1.5,0.5])
        with col5:
            st.header('Variables globales',divider='gray')

    response_explain = requests.get(f"{api_base_url}/explain")
    if response_explain.status_code == 200:
        col1,col2,col3=st.columns([0.5,1.5,0.5])
        with col2: 
            st.image((response_explain.text).strip('"'))
    else:
        st.error(f"Failed to generate feature importance plot. Error: {response_explain.text}")

    col1,col2,col3=st.columns([0.5,1.5,0.5])
    with col2:
        col4,col5,col6=st.columns([0.5,1.5,0.5])
        with col5:
            st.header('Impact des variables pour les IDs sélectionnés',divider='gray')

        st.subheader('Une évolution :red[rouge] indique une variable influant :red[positivement] pour l\'acceptation des prêts')
        st.subheader('Une évolution :blue[bleue] indique une variable influant :blue[négativement] pour l\'acceptation des prêts')
        st.subheader('')

    for selected_value in selected_values:

        response_id_explain = requests.get(f"{api_base_url}/explain/{selected_value}")
        if response_id_explain.status_code == 200:
            col1,col2,col3=st.columns([0.5,1.5,0.5])
            with col2: 
                st.image((response_id_explain.text).strip('"'))
        else:
            st.error(f"Failed to generate waterfall plot. Error: {response_id_explain.text}")

# Section Rapport Data Drift
col1,col2,col3=st.columns([0.5,1.5,0.5])
with col2: 
    st.header("Rapport du Data Drift de la base de données",divider="orange")

col1,col2=st.columns([0.5,2])
with col2:
     
    if response_data_drift.status_code == 200:
        html_content = response_data_drift.text
        components.html(html_content, width=1600, height=1000, scrolling=True)
    else:
        st.error(f"Failed to fetch HTML file. Status code: {response_data_drift.status_code}")

