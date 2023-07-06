from flask import Flask, request, jsonify
from time import sleep
import pandas as pd
import numpy as np
from io import BytesIO
from matplotlib import pyplot
import base64
import pickle
import shap
import requests
import os

def create_app(config={"TESTING": False}):
    api = Flask(__name__)
    api.config.from_object(config)

    
    api_initialized = False
    model = None
    X = None
    
    threshold = 0.49

    print("Initialisation débutée")
    model = pickle.load(open("xgb_1/model.pkl", "rb"))

    if not os.path.isdir("assets"):
        os.mkdir("assets")

    if not os.path.exists("assets/df_application_test.parquet"):
        LOGIN = os.environ.get("parquet_get_login")
        assert LOGIN is not None
        PASSWORD = os.environ.get("parquet_get_password")
        assert PASSWORD is not None
        URL = os.environ.get("parquet_get_url")
        assert URL is not None
        print("Téléchargement du fichier parquet")
        response = requests.get(url = URL, auth=(LOGIN,PASSWORD))
        with open("assets/df_application_test.parquet", "wb") as parquet_file:
            parquet_file.write(response.content)
        print("Fichier téléchargé")

    X = pd.read_parquet("assets/df_application_test.parquet")

    explainer = shap.TreeExplainer(model, max_evals=1000, feature_names=X.drop(columns="SK_ID_CURR").columns)

    api_initialized = True
    print("Initialisation terminée")

    
    def get_explanation(sk_id_curr, max_display=50, return_base64=True, show_plot=False):
        if X is None:
            return {
                "success": False,
                "message": f"Données non chargées"
            }
            
        ind = X.loc[X["SK_ID_CURR"]==sk_id_curr].index
        if len(ind)==0:
            return {
                "success": False,
                "message": f"SK_ID_CURR {sk_id_curr} non trouvé"
            }
    
        ind = ind[0]
        X_shap = np.array(X.iloc[ind:ind+1].drop(columns="SK_ID_CURR"), dtype=float)
        shap_values = explainer(X_shap)
    
        if show_plot:
            shap.plots.waterfall(shap_values[0], max_display=max_display, show=True)
            
        if return_base64:
            pyplot.clf()
            shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
            image = BytesIO()
            pyplot.savefig(image, format='png', bbox_inches='tight')
            return {
                "success": True,
                "image": base64.encodebytes(image.getvalue()).decode('utf-8')
            }


    @api.route("/health")
    def hello():
        return f"api_initialized={api_initialized}"

    
    @api.route('/predict/<sk_id_curr>', methods = ['GET'])
    def predict(sk_id_curr):
        if api_initialized==False:
            return {
                "success": False,
                "message": "API non intialisée"
            }
            
        if sk_id_curr.strip()=="":
            return {
                "success": False,
                "message": "SK_ID_CURR non renseigné"
            }
            
        if not sk_id_curr.isdigit():
            return {
                "success": False,
                "message": "SK_ID_CURR n'est pas un entier naturel"
            }
    
        sk_id_curr = int(sk_id_curr)
    
        max_display = int(request.args.get('max_display'))
    
        explanation = get_explanation(sk_id_curr,return_base64=True, show_plot=False, max_display=max_display)
        if explanation["success"]==False:
            return explanation
    
        
        proba = model.predict_proba( X.loc[X["SK_ID_CURR"]==sk_id_curr].drop(columns="SK_ID_CURR") )[0]
        if proba[1]>threshold:
            explanation["conclusion"] = 1
        else:
            explanation["conclusion"] = 0
    
        explanation["conclusion_proba"] = [np.float64(proba[0]), np.float64(proba[1])]
        return jsonify(explanation)

    return api


if __name__ == "__main__":
    api = create_app({"TESTING": False})
    api.run(host='0.0.0.0', port=12080)
