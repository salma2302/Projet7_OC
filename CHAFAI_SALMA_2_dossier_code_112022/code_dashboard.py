import streamlit as st
import json
import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
import shap
import numpy as np
import pickle
import dill

#Application hello word
st.title("Prédiction de l'accord d'un crédit d'un client ")
#st.subheader("Données des cliens")
st.markdown(":tada: Cette application affiche si on accorde ou non un prêt à un client ")


#--------------image-----------------------------------------
img = Image.open("photo_entreprise.png")
st.image(img, width=200)




# Chargement du jeu de données servi pout l'entrainement (finir cette partie)
#train = pd.read_csv('df_train.csv')

# Choix de l'identifiant
st.sidebar.header("Choix du client")
res_id = requests.get("http://127.0.0.1:8000/clients")
id_client = res_id.json()["id_clients"]
id_selected = st.sidebar.selectbox("Choix de l'id du client", options=id_client)

# La liste déroulante des features
#option_features = st.sidebar.selectbox('Sélection ta variable 1',tuple(liste_cols))
# Appeler la route qui retourne la liste des colonnes
res_col = requests.get("http://127.0.0.1:8000/columns")
res_description = requests.get("http://127.0.0.1:8000/description_columns")
columns = res_col.json()["columns"]
description = res_description.json()["description_columns"]


#ligne = df[df['SK_ID_CURR'] == id_selected]

st.sidebar.header("Choix des variables à afficher")

# On crée la première liste déroulante
columns_selected1 = st.sidebar.selectbox('Sélectionnez la première variable :', options=columns)
columns_disponibles = columns.copy()

# Enlever la variable sélectionnée de la liste des variables disponibles
columns_disponibles.remove(columns_selected1)

# Utiliser la liste des variables disponibles pour afficher la deuxième liste déroulante
columns_selected2 = st.sidebar.selectbox('Sélectionnez la deuxième variable :', options=columns_disponibles)

# Afficher les variables sélectionnées
st.write('Vous avez sélectionné les variables suivantes :', columns_selected1, 'et', columns_selected2)


# L'inputs 
inputs = {"id_client" : id_selected}

#---------------------------------------------
## Les infos du clients 
response_client = requests.get(f"http://127.0.0.1:8000/client/{id_selected}")

def formatage_date(chaine):
    date_ref = datetime.datetime(1970, 1, 1)
    date = date_ref + datetime.timedelta(days=chaine)
    
    date_f = date.strftime('%Y-%m-%d')
    
    return date_f


# vérifier si la réponse est valide
if response_client.status_code == 200:
    # récupérer les données du client sous forme de dictionnaire
    client_data = response_client.json()[0]
    
    client_data = pd.DataFrame.from_dict(client_data, orient='index').transpose()
    #st.write(client_data)
else:
    st.error("Error getting client data")
    
#---------------------------------------------




if st.button("Informations_client") :
    st.subheader('Voici les infos du client')
    affichage = client_data.copy()
    affichage['DAYS_BIRTH'] = affichage['DAYS_BIRTH'].apply(lambda x : formatage_date(x))
    affichage['DAYS_EMPLOYED'] = affichage['DAYS_EMPLOYED'].apply(lambda x : formatage_date(x))
    affichage = affichage.T.reset_index()
    affichage.columns = ['Nom_variable', 'Valeur']
    affichage['description'] = description
    st.table(affichage)

st.sidebar.header("Graphiques")

if st.sidebar.button("Graphique univarié") :
    
    response1 = requests.get(f"http://127.0.0.1:8000/column/{columns_selected1}")
    response2 = requests.get(f"http://127.0.0.1:8000/column/{columns_selected2}")
    
    if response1.status_code == 200 and response2.status_code == 200:
        # Récupérer les données de la réponse
        df1 = response1.json()
        df2 = response2.json()
        
        point = client_data[[columns_selected1, columns_selected2]]
        
        ## Graphique pour la variable 1
        
        # Créez les deux graphes avec Plotly
        box1 = go.Box(x=df1['column_values'], name='Tous les clients')
          
        # Créez un trace avec les données du client
        trace1 = go.Scatter(x=point[columns_selected1], y=[0], mode='markers', name='Client', marker=dict(color='#e7298a', size=10))

        
        fig_var1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_var1.add_trace(box1)
        fig_var1.add_trace(trace1,secondary_y=True)

        
        fig_var1.update_layout(
            title=f"Plot du boxplot de la variable {columns_selected1}",
            xaxis_title= f"La variable {columns_selected1}",
            legend_title="Legend Title"
)
        
        
        ## Graphique pour la variable 2
        
        # Créez les deux graphes avec Plotly
        box2 = go.Box(x=df2['column_values'], name='Tous les clients')
          
        # Créez un trace avec les données du client
        trace2 = go.Scatter(x=point[columns_selected2], y=[0], mode='markers', name='Client', marker=dict(color='#e7298a', size=10))

        
        fig_var2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_var2.add_trace(box2)
        fig_var2.add_trace(trace1,secondary_y=True)

        
        fig_var2.update_layout(
            title=f"Plot du boxplot de la variable {columns_selected2}",
            xaxis_title= f"La variable {columns_selected2}",
            legend_title="Legend Title"
)
        
        






        # Affichez les figures côte à côte
        st.plotly_chart(fig_var1)
        st.plotly_chart(fig_var2)
        
    
    


# On affiche les graphique bivariée des deux variables choisies
if st.sidebar.button("Graphique bivariée") :
    
    response1 = requests.get(f"http://127.0.0.1:8000/column/{columns_selected1}")
    response2 = requests.get(f"http://127.0.0.1:8000/column/{columns_selected2}")

    # Afficher les colonnes dans l'interface utilisateur
    # Vérifier le code de retour de la réponse
    if response1.status_code == 200 and response2.status_code == 200:
        # Récupérer les données de la réponse
        data1 = response1.json()
        data2 = response2.json()
        
        point = client_data[[columns_selected1, columns_selected2]]
        
        # Créer un graphique à barres en utilisant plotly
        fig = px.scatter(x=data1['column_values'], y=data2['column_values'],
                         color=data1["target"],
                         color_discrete_sequence=["green", "red"]
        )
        fig.add_scatter(x=point[columns_selected1], y=point[columns_selected2],
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                       name='Client')
        
        fig.update_layout(
            title=f"Plot du boxplot de la variable {columns_selected1} en fonction de la variable {columns_selected2}",
            xaxis_title= f"La variable {columns_selected1}",
            yaxis_title= f"La variable {columns_selected2}",
            legend_title="Legend Title"
)
        
        # Afficher le graphique dans l'interface utilisateur
        st.plotly_chart(fig, use_container_width=True)
        



    

st.sidebar.header("Prédiction")

if st.sidebar.button("predict") :
    res = requests.post(url = "http://127.0.0.1:8000/predict", params=inputs)
    prediction = res.json()
    
    
    st.success(f"Le crédit est {prediction['prediction']} avec une proba de {prediction['probabilité']} d'appartenance à la classe 0 ('solvable') pour le client avec l'id {id_selected}")
    
       





        
        
# A revoir        
if st.sidebar.button("Features importance") :
    
    # Chargement du de l'explainer
    shap_explainer = dill.load(open("shap_explainer.dill","rb"))


    response = requests.get("http://127.0.0.1:8000/get_X_test")
    if response.status_code == 200:
        
        X_test = response.json()["data"] 
        X_test = pd.DataFrame(X_test)
        

        shap_values = shap_explainer.shap_values(X_test)

        st.subheader("Valeurs SHAP")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        st.subheader("Interprétabilité globale")
        st.pyplot(shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist()))
        
        st.subheader(f"Interprétabilité globale : Pour le client {id_selected}")
        shap_val = shap_explainer.shap_values(client_data)
        # Créer un trace Plotly avec les données de l'explication SHAP

        # Afficher le trace avec st.plotly_chart
        shap.plots._waterfall.waterfall_legacy(shap_explainer.expected_value,
                                               shap_val[0],
                                               feature_names = client_data.columns,
                                                max_display= 10) 
        st.pyplot()
        

        
    else :
        st.write("Erreur: la requête a échoué avec le code d'état", response.status_code)
    



         
         
