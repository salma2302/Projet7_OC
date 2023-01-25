# 1. Library imports
# Import Needed Libraries
import joblib
import uvicorn
import numpy as np
import pandas as pd
import pickle
import dill
from fastapi import FastAPI

# 2. Create the app object
app = FastAPI()

# Initialize model artifacte files. This will be loaded at the start of FastAPI model server.
#pickle_in = open("model_credit_fr.pkl","rb")
#classifier=pickle.load(pickle_in)
features_selected=pickle.load(open("features_selected.pkl","rb"))
features_description=pickle.load(open("features_description.pkl","rb"))


@app.get("/")
async def root():
    return {"message": "Bienvenue dans l'Api d'accord de crédit"}


#--------------------------------------------------------------------------------
# Chargement du modèle et du jeu de données 
@app.get("/data")
def get_dataframe():
    # sélectionner les colonnes spécifiées du DataFrame
    # Chargement d'un échantillon de 100 clients
    df = pd.read_csv('echantillon.csv')
    features_selected=pickle.load(open("features_selected.pkl","rb"))
    

    
    # convertir le DataFrame en dictionnaire
    df = df.to_dict(orient="records")
    
    resultat = {"data": df}
    
    return resultat


@app.get("/model")
def get_model():
    
    pickle_in = open("model_credit.pkl","rb")
    classifier=pickle.load(pickle_in)
    
    return {"model": classifier}

#---------------------------------------------------------------------------------------------------

# Le jeu de données
df = pd.DataFrame(get_dataframe()['data'])


# Récupérer la liste des colonnes du jeu de données
columns = features_selected





# Récupérer la liste des id_clients
l_id_client = df['SK_ID_CURR'].tolist()

# Définir une route qui retourne la liste des id clients
@app.get("/clients")
def ids_route():
    return {"id_clients": l_id_client}

@app.get("/client/{id_client}")
def get_client(id_client: int):
    # récupérer les données de la base de données en utilisant l'id_client
    
    X = df[df["SK_ID_CURR"] == id_client]
    X = X[features_selected]
    
    
    # vérifier si les données ont été trouvées
    if X.empty:
        raise HTTPException(status_code=404, detail=f"Client with id_client {id_client} not found")
    
    # renvoyer les données sous forme de dictionnaire
    return X.to_dict(orient="records")

# Définir une route qui retourne la liste des colonnes
@app.get("/columns")
def columns_route():
    return {"columns": columns}

@app.get("/description_columns")
def columns_route():
    return {"description_columns": features_description}

@app.get("/column/{column_name}")
def column_route(column_name: str):
    
    


    # Récupérer la valeur de la colonne
    column_values = df[column_name].values.tolist()

    # La target
    y = df["TARGET"].values.tolist()
    # liste de chaînes de caractères correspondantes
    labels = ["solvable","non_solvable"]
    
    y = list(map(lambda x: labels[x], y))
    return {"column_name": column_name, "column_values": column_values, "target" : y}



# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.post('/predict')
def predict(id_client: int):
    # On récupère les informations du client
    
    ligne = df[df['SK_ID_CURR'] == id_client]
    X_test = ligne[features_selected]
    
    classifier = get_model()['model']
    
    # Prédire la probabilité pour le client
    pred_proba = classifier.predict_proba(X_test)

    # Le seuil choisit avec la fonction personnalisée
    seuil_optimal = 0.62
    
    y_pred = classifier.predict(X_test)
    y_pred[pred_proba[:,1] > seuil_optimal] = 1
    y_pred[pred_proba[:,1] <= seuil_optimal] = 0
    
    # La prédiction en fonction du seuil
    prediction_label = ["accordé" if y_pred == 0 else "refusé"]
    
    
    
    # Return la prédiction au client avec la proba associé
    return {"prediction": prediction_label[0],
           "probabilité" : max(pred_proba[0]),
           "seuil_optimal" : seuil_optimal}


@app.get("/get_X_test")
def get_X_test():
    # sélectionner les colonnes spécifiées du DataFrame
    # Chargement d'un échantillon de 100 clients
    
   
    
    cols = features_selected
    df_selected = df[cols]
    
    
    
    # convertir le DataFrame en dictionnaire
    data = df_selected.to_dict(orient="records")
    
    return {"data": data}











if __name__ == '__main__':
    uvicorn.run("app_credit:app", host='0.0.0.0', port=80)
