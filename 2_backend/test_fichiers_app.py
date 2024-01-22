import pandas as pd
import os
import joblib
from lightgbm import LGBMClassifier
import shap


###########################################################################
#######################    Tests sur le fichier 1   #######################
###########################################################################

file_path_1='API_input_data/Liste_index_clients_futur.csv'

exist_1=os.path.exists(file_path_1)

# Existence du fichier
def test_index_clients_existe():
    assert exist_1==True

if exist_1==True:

    file_1=pd.read_csv(file_path_1,index_col='ID')

    # Fichier est un dataframe
    def test_index_clients_format_df():
        assert isinstance(file_1,pd.DataFrame)==True

    if isinstance(file_1,pd.DataFrame)==True:

        # Dataframe d'une seule colonne
        def test_index_client_colonne():
            assert file_1.shape[1]==1

        # Les index sont uniques
        def test_index_clients_unique():
            assert len(file_1.index.unique())==file_1.shape[0]

###########################################################################
#######################    Tests sur le fichier 2   #######################
###########################################################################

file_path_2='API_input_data/base_futur.csv'

exist_2=os.path.exists(file_path_2)

def test_base_existe():
    assert exist_2==True    

if exist_2==True:

    file_2=pd.read_csv(file_path_2,sep='\t',compression='gzip',index_col='ID')

    # Fichier est un dataframe
    def test_base_format_df():
        assert isinstance(file_2,pd.DataFrame)==True

    if isinstance(file_2,pd.DataFrame)==True:

        # Base de longueur index clients
        def test_base_egal_index():
            assert file_2.shape[0]==file_1.shape[0]

###########################################################################
#######################    Tests sur le fichier 3   #######################
###########################################################################
            
file_path_3='API_input_data/model_risk_100.joblib'

exist_3=os.path.exists(file_path_3)

# Existence du fichier
def test_modele_existe():
    assert exist_3==True

if exist_3==True:

    file_3=joblib.load(file_path_3)

    # Fichier est un modele lgbm après lecture par joblib
    def test_modele_format_job():
        assert isinstance(file_3,LGBMClassifier)==True

###########################################################################
#######################    Tests sur le fichier 4   #######################
###########################################################################
        
file_path_4='API_input_data/feat_important_sorted_decroissant.csv'

exist_4=os.path.exists(file_path_4)

# Existence du fichier
def test_features_existe():
    assert exist_4==True

if exist_4==True:

    file_4=pd.read_csv(file_path_4,header=None)

    # Fichier est un dataframe
    def test_features_format_df():
        assert isinstance(file_4,pd.DataFrame)==True

###########################################################################
#######################    Tests sur le fichier 5   #######################
###########################################################################

file_path_5='API_input_data/report_5.html'

exist_5=os.path.exists(file_path_5)

# Existence du fichier
def test_report_existe():
    assert exist_5==True

if exist_5==True:

    with open(file_path_5, encoding='utf-8') as file:
        file_5=file.read()

    # Fichier ouvert est lisible comme string => simplification de considérer équivalent à du html
    def test_report_format_html():
        assert isinstance(file_5,str)==True

###########################################################################
#######################    Tests sur le fichier 6   #######################
###########################################################################
        
file_path_6='API_input_data/base_futur_prep_algo.csv'

exist_6=os.path.exists(file_path_6)

# Existence du fichier
def test_base_algo_existe():
    assert exist_6==True

if exist_6==True:

    file_6=pd.read_csv(file_path_6,sep='\t',compression='gzip',index_col='ID')

    # Fichier est un dataframe
    def test_base_algo_df():
        assert isinstance(file_6,pd.DataFrame)==True

    if isinstance(file_6,pd.DataFrame)==True:

        # Base de même longueur que l'index clients
        def test_base_algo_egal_index():
            assert file_6.shape[0]==file_1.shape[0]
    
###########################################################################
#######################    Tests sur le fichier 7   #######################
###########################################################################

file_path_7='API_input_data/shap_explanation_model_risk_100.joblib'

exist_7=os.path.exists(file_path_7)

# Existence du fichier
def test_shap_existe():
    assert exist_7==True

if exist_7==True:

    file_7=joblib.load(file_path_7)

    # Fichier est un objet shap Explanation
    def test_shap_format_explanation():
        assert isinstance(file_7,shap.Explanation)==True

###########################################################################
#######################    Tests sur le fichier 8   #######################
###########################################################################

file_path_8='API_input_data/shap_waterfall_plot.jpeg'

exist_8=os.path.exists(file_path_8)

# Existence du fichier
def test_waterfall_existe():
    assert exist_8==True

if exist_8==True:

    with open(file_path_8, 'rb') as file:
        header_8 = file.read(2)

    # Le header extrait de l'ouverture du fichier indique qu'il s'agit d'un jpeg
    def test_waterfall_format_jpeg():
        assert header_8 == b'\xff\xd8'

###########################################################################
#######################    Tests sur le fichier 9   #######################
###########################################################################

file_path_9='API_input_data/lgbm_features_plot.jpeg'

exist_9=os.path.exists(file_path_9)

# Existence du fichier
def test_feat_plot_existe():
    assert exist_9==True

if exist_9==True:

    with open(file_path_9, 'rb') as file:
        header_9 = file.read(2)

    # Le header extrait de l'ouverture du fichier indique qu'il s'agit d'un jpeg
    def test_feat_plot_format_jpeg():
        assert header_9 == b'\xff\xd8'






    


    