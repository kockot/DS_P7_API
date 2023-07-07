from flask import Flask, request, jsonify, render_template
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
from flask_wtf.csrf import CSRFProtect

def create_app(config={"TESTING": False, "TEMPLATES_AUTO_RELOAD": True}):
    api = Flask(__name__)

    api.config.from_object(config)

    SECRET_KEY = os.urandom(32)
    api.config['SECRET_KEY'] = SECRET_KEY
    api.config['WTF_CSRF_SECRET_KEY'] = SECRET_KEY

    csrf = CSRFProtect()
    csrf.init_app(api)

    api_initialized = False
    model = None
    X = None
    
    threshold = 0.49

    print("Initialisation débutée")
    model = pickle.load(open("xgb_1/model.pkl", "rb"))

    SECURITY_TOKEN = os.environ.get("security_token")
    assert SECURITY_TOKEN is not None

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


    def get_prediction(sk_id_curr, max_display):
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


    @api.route("/health")
    @csrf.exempt
    def health():
        return f"api_initialized={api_initialized}"


    @api.route("/application")
    def application():
        return render_template('search_form.html')

    @api.route("/sk_id_curr", methods = ["GET"])
    def list_sk_id_curr():
        if api_initialized==False:
            return {
                "success": False,
                "message": "API non intialisée"
            }
        
        return {
            "success": True,
            "data": X.loc[:, "SK_ID_CURR"].to_list()
        }


    @api.route('/predict/<sk_id_curr>', methods = ['GET'])
    @csrf.exempt
    def predict(sk_id_curr):
        if api_initialized==False:
            return {
                "success": False,
                "message": "API non intialisée"
            }
            
        headers = request.headers
        bearer = headers.get('Authorization')    # Bearer YourTokenHere
        if bearer is None:
            return {
                "success": False,
                "message": "Jeton d'authentification non fourni"
            }

        token = bearer.split()[1]
        if token is None or token!=SECURITY_TOKEN:
            return {
                "success": False,
                "message": "Echec de l'authentification du jeton"
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
        if request.args.get('max_display') is not None:
            max_display = int(request.args.get('max_display'))
        else:
            max_display = 25
            
        return get_prediction(sk_id_curr, max_display)


    @api.route('/predict2/<sk_id_curr>', methods = ['POST'])
    def predict_POST(sk_id_curr):
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
        data = request.json
        if data.get('max_display') is not None:
            max_display = int(data.get('max_display'))
        else:
            max_display = 25
            
        return get_prediction(sk_id_curr, max_display)


    return api




if __name__ == "__main__":
    api = create_app({"TESTING": False})
    api.run(host='0.0.0.0', port=8000)
