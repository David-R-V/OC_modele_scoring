# Imports de librairies
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import csv
import gzip
from itertools import islice
import seaborn as sns
from PIL import Image
from scipy import stats

# Définitions diverses
feat_important_sorted_decroissant=pd.Series(pd.read_csv('../1_outputs/feat_important_sorted_decroissant.csv',header=None)[0])
feat_important_sorted_decroissant=feat_important_sorted_decroissant.array
var_lgbm_100=np.append(feat_important_sorted_decroissant[:100],'TARGET')
info_base=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
            'AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED',
            'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']
selection_var=info_base + [i for i in var_lgbm_100 if i not in info_base]
serie_index_clients = pd.read_csv("../1_outputs/Liste_index_clients_futur.csv",index_col='ID')

# Stockage d'une sélection aléatoire d'index clients à requeter
if 'random_sel' not in st.session_state:
    st.session_state.random_sel = random.sample(list(serie_index_clients.index), 15)

# Fonction d'extraction des données depuis le df pour le dashboard 
# (adapté sur la base d'un csv source énorme : extraction uniquement de la ligne voulue sans lire le fichier entier)
def get_pf_data(index_cible):

    file_path = "../1_outputs/base_futur.csv"
    separator = '\t'
    compression = 'gzip'
    line_number=serie_index_clients.index.get_loc(index_cible)+1

    with gzip.open(file_path,mode='rt',encoding='latin1') as file:
        selected_line = next(islice(file, line_number , line_number+1), None)

    if selected_line is not None:
        if selected_line.endswith('\n') and not selected_line.strip().endswith(','):
            selected_line = selected_line.rstrip('\n')

        selected_line=pd.DataFrame(selected_line.split(separator),
                                   index=pd.read_csv(file_path,sep=separator,compression=compression,nrows=1).columns.to_list()).T.set_index('ID')
        selected_line=selected_line.replace('',np.nan)
        return selected_line[info_base].round(3)
    else: 
        return None

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

# Boite du multiselect de l'index client
selected_values = st.multiselect("Sélection de l'index ID:", st.session_state.random_sel)

# Limitation du nombre de sélections simultanées à 3
max_selections = 3
if len(selected_values) > max_selections:
    st.warning(f"Maximum de {max_selections} sélections affichées atteint.")
    selected_values = selected_values[:max_selections] 


# Sélection des données affichées lorsque sélectionnées dans le multiselect 1
temp=pd.DataFrame(columns=info_base)
if selected_values:
    for selected_value in selected_values:
        temp =pd.concat([temp, get_pf_data(selected_value)])
    st.header("Informations générales",divider="orange")
    temp=temp.T
    
    st.dataframe(temp,width=600)
else:
    st.warning("No values selected.")


df_infos=pd.read_csv("../1_outputs/base_futur.csv",sep='\t',compression='gzip',index_col='ID')[info_base]

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


if selected_values:
    st.header("Répartitions sur variables catégorielles",divider="orange")
    for selected_value in selected_values:
        col1,col2,col3,col4,col5 = st.columns([1,1.3,1,1.3,1])
        for col_x,col_y in zip(['CODE_GENDER', 'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT'],[col1,col3,col5]):
            with col_y:            
                st.image(img_pieplot(selected_value,col_x,col_x),use_column_width=False)    
    
else:
    st.warning("No values selected.")

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
    

def img_kdeplot(selected_value,col,nom_fichier):
    
    kde_clip(selected_value,col,nom_fichier)
    image = Image.open(f'{nom_fichier}.png')

    return image

if selected_values:
    st.header("Distributions des variables numériques",divider="orange")
    for selected_value in selected_values:
        col1,col2,col3,col4 = st.columns([0.07,1,0.5,1])
        for col_x,col_y in zip(['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE'],[col2,col4]):
            with col_y:            
                st.image(img_kdeplot(selected_value,col_x,col_x),use_column_width=False)



# col_value_counts=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN','FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT',
#        'REGION_RATING_CLIENT_W_CITY']
# col_numeric=['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY','REGION_POPULATION_RELATIVE','AMT_GOODS_PRICE','DAYS_BIRTH', 'DAYS_EMPLOYED']


