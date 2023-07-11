from flask import Flask, request, jsonify, render_template
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
import json
import orjson
def get_df_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    signs = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
        if shap_values.values[:, i]==0:
            signs.append(0)
        elif shap_values.values[:, i]>0:
            signs.append(1)
        else:
            signs.append(-1)

    df = pd.DataFrame({"feature_name": features, "importance": importances, "sign": signs})
    df.sort_values("importance", ascending=False, inplace=True)
    return df


def create_app(config={"TESTING": False, "TEMPLATES_AUTO_RELOAD": True}):
    api = Flask(__name__)

    api.config.from_object(config)

    SECRET_KEY = os.urandom(32)
    api.config['SECRET_KEY'] = SECRET_KEY
    api.config['WTF_CSRF_SECRET_KEY'] = SECRET_KEY

    api.config["TEMPLATES_AUTO_RELOAD"] = True

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


    X = pd.read_parquet("assets/df_application_test.parquet", filters=[("SK_ID_CURR", "=", 100001)])
    explainer = shap.TreeExplainer(model, max_evals=1000, feature_names=X.drop(columns="SK_ID_CURR").columns)

    X = pd.read_parquet("assets/df_application_test.parquet", columns=["SK_ID_CURR"])
    arr_sk_id_curr = X["SK_ID_CURR"].to_list()
    del X

    api_initialized = True
    print("Initialisation terminée")

    
    def get_explanation(X, sk_id_curr, max_display=50, return_base64=True, show_plot=False):
        if not api_initialized:
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
        vals = np.abs(shap_values.values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(X.columns, vals)),
            columns=['col_name', 'feature_importance_vals']
        )
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)


        if show_plot:
            shap.plots.waterfall(shap_values[0], max_display=max_display, show=True)
            
        if return_base64:
            pyplot.clf()
            shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
            image = BytesIO()
            pyplot.savefig(image, format='png', bbox_inches='tight')
            ret = {
                "success": True,
                "image": base64.encodebytes(image.getvalue()).decode('utf-8'),
                "features_importances": []
            }

            # let s continue with the features distributions
            df_feat_impo = get_df_feature_importances_shap_values(shap_values, X.drop(columns="SK_ID_CURR").columns)
            for i in range(0, max_display):
                line = df_feat_impo.iloc[i]
                fname = f"""assets/hist_json/{line["feature_name"].replace("/", "_")}.json"""
                with open(fname) as json_f:
                    decoded = json.load(json_f)
                    ret["features_importances"].append({
                        "name": str(line["feature_name"]),
                        "value": float(X.loc[ind, line["feature_name"]]),
                        "contribution": float(line["importance"] * line["sign"]),
                        "hist_json_y": list(float(y) for y in decoded[0]),
                        "hist_json_x": list(float(x) for x in decoded[1]),
                    })
            return ret

    def get_prediction(X, sk_id_curr, max_display):
        explanation = get_explanation(X, sk_id_curr,return_base64=True, show_plot=False, max_display=max_display)
        if explanation["success"]==False:
            return explanation
    
        
        proba = model.predict_proba( X.loc[X["SK_ID_CURR"]==sk_id_curr].drop(columns="SK_ID_CURR") )[0]
        if proba[1]>threshold:
            explanation["conclusion"] = 1
        else:
            explanation["conclusion"] = 0
    
        explanation["conclusion_proba"] = [float(proba[0]), float(proba[1])]
        return orjson.dumps(explanation)


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
            "data": arr_sk_id_curr
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
            
        X = pd.read_parquet("assets/df_application_test.parquet", filters=[("SK_ID_CURR", "=", sk_id_curr)])
        return get_prediction(X, sk_id_curr, max_display)


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

        X = pd.read_parquet("assets/df_application_test.parquet", filters=[("SK_ID_CURR", "=", sk_id_curr)])
        return get_prediction(X, sk_id_curr, max_display)


    return api




if __name__ == "__main__":
    api = create_app({"TESTING": False})
    api.run(host='0.0.0.0', port=8000)
