# Introduction

Ce projet est un exemple d'API construite autour du modèle retenu dans le [projet 7](../DataScience_Training/tree/main/projet7) .

Il nécessite les variables d'environnement suivantes:
- security_token : code de sécurité partagé avec le frontend StreamLit
- parquet_get_url : URL du fichier des données de production (df_application_test.parquet)
- parquet_get_login : login pour récupérer le fichier parquet
- parquet_get_password : mot de passe pour récupérer le fichier parquet

# Services fournis

Il fournit les services sous forme de endpoints web:
- /health (GET): retourne l'état de l'initialisation de l'API. Quand l'API est prête: **api_initialized=True**
- /sk_id_curr (GET): retourne le tableau de tous les identifiants SK_ID_CURR du dataset de production au format JSON
- /predict/<sk_id_curr> (GET): retourne les informations de prédictions concernant la demande identifiée par \<sk_id_curr\> dans l'adresse fournie.
  En entrée, sont attendus:
  - dans l'entête de la requête, une authentification via une ligne
  > **'Authorization': 'Bearer {security_token}'**
  > avec {security_token} à remplacer par la valeur de la variable d'environnement security_token
  >
  - optionnellement, en tant que paramètre HTTP GET:
  > **paramètre max_display en option**
  >
  > permet de définir le nombre de features les plus importantes à expliquer (25 par défaut)
    
  Les informations retournées sont fournies en tant que dictionnaire au format JSON, contenant toujours une entrée avec la clé "success"
    - en cas d'erreur, l'entrée "success" aura la valeur False et une entrée avec la clé "message" fournira une information sur l'erreur
    - sinon,
      
    > {
    >
    >      "success": True,
    >      "conclusion": int(0=demande acceptée, 1=demande rejetée)
    >      "conclusion_proba": array(float(pourcentage de probabilité de crédit accepté), float(pourcentage de probabilité de crédit refusé))
    >      "image_global": str(explication globale du modèle au format png encodée en base64)
    >      "image": str(explication locale sous forme de waterfall au format png encodée en base64)
    >      "features_importances": array({
    >          "name": str(nom de la feature),
    >          "value": float(valeur de la feature pour  l'observation demandée),
    >          "description": str(description de la feature),
    >          "contribution": float(contribution de la feature à l'explication locale),
    >          "hist_json_y": array(float(distribution de histogramme de répartition de la feature au niveau global selon l'axe y))
    >          "hist_json_x": array(float(distribution de histogramme de répartition de la feature au niveau global selon l'axe x))
    > 
    > }
- /login (POST): permet d'obtenir un token JWT pour authentifier un utilisateur
  En entrée sont attendus les paramètres:
    - username: str(nom d'utilisateur)
    - password: str(mot de passe)
  Les informations retournées sont fournies en tant que dictionnaire au format JSON, contenant toujours une entrée avec la clé "success"
    - en cas d'erreur, l'entrée "success" aura la valeur False et une entrée avec la clé "message" fournira une information sur l'erreur
    - sinon,
      
    > {
    >
    >      "success": True,
    >      "access_token": str(token JWT)
    > }
- /predict2/<sk_id_curr> (GET): retourne les informations de prédictions concernant la demande identifiée par \<sk_id_curr\> dans l'adresse fournie.
  En entrée, sont attendus:
  - dans l'entête de la requête, une authentification via un jeton JWT préalablement obtenu via l'appel à /login (POST)
  > **'Authorization': 'Bearer {jwt_token}'**
  > avec {jwt_token} à remplacer par la valeur du token JWT
  >
  - optionnellement, en tant que paramètre HTTP GET:
  > **paramètre max_display en option**
  >
  > permet de définir le nombre de features les plus importantes à expliquer (25 par défaut)
    
  Les informations retournées sont fournies en tant que dictionnaire au format JSON, contenant toujours une entrée avec la clé "success"
    - en cas d'erreur, l'entrée "success" aura la valeur False et une entrée avec la clé "message" fournira une information sur l'erreur
    - sinon,
      
    > {
    >
    >      "success": True,
    >      "conclusion": int(0=demande acceptée, 1=demande rejetée)
    >      "conclusion_proba": array(float(pourcentage de probabilité de crédit accepté), float(pourcentage de probabilité de crédit refusé))
    >      "image_global": str(explication globale du modèle au format png encodée en base64)
    >      "image": str(explication locale sous forme de waterfall au format png encodée en base64)
    >      "features_importances": array({
    >          "name": str(nom de la feature),
    >          "value": float(valeur de la feature pour  l'observation demandée),
    >          "description": str(description de la feature),
    >          "contribution": float(contribution de la feature à l'explication locale),
    >          "hist_json_y": array(float(distribution de histogramme de répartition de la feature au niveau global selon l'axe y))
    >          "hist_json_x": array(float(distribution de histogramme de répartition de la feature au niveau global selon l'axe x))
    > 
    > }
      
# Installation des prérequis
```
pip3 install -r requirements.txt
```

# Tests unitaires
```
pip install -r requirements.txt
export security_token=whatever_you_want
pytest -v tests/
```


# Démarrage de l'API
```
pip install -r requirements.txt

# Next is the security token shared with the Streamlit frontend ap

export security_token=xxx

# Next are the info to get the parquet dataset.
# Due to Heroku's limitation on app bundling (500M), the application fetches external
# needed data and installs it locally

export parquet_get_url=**provided url**
export parquet_get_login=**provided login**
export parquet_get_password=**provided password**

python3 app.py
```

















