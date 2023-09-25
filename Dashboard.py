# %% [markdown]
# # Dashboard

# %%
import streamlit as st
import pandas as pd
import numpy as np

# %%
st.title("Application de prédiction de risque de défaut individuel")

# %%
feat_important_sorted_decroissant=pd.Series(pd.read_csv('Outputs/feat_important_sorted_decroissant.csv',header=None)[0])
feat_important_sorted_decroissant=feat_important_sorted_decroissant.array
var_lgbm_100=np.append(feat_important_sorted_decroissant[:100],'TARGET')

# %%
@st.cache_data
def get_pf_data():
    
    df = pd.read_csv("Outputs/df_total.csv",sep='\t',compression='zip',index_col='ID')
    return df

# %%
df = get_pf_data()

# %%
info_base=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
 'AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED',
 'FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']

# %%
#print(pd.DataFrame(data=df.columns).to_markdown())

# %%
client = st.multiselect(
    "Choix de l'ID Client à afficher", list(df.index),
)
if not client:
    st.error("Au moins une ID doit être sélectionnée")
else:
    donnee_affichee = df.loc[client,info_base].T
    st.write("### Informations générales", donnee_affichee)


# %%
df[['CODE_GENDER','FLAG_OWN_CAR']].head()

# %%



