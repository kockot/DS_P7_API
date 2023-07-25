from flask import Flask, request, jsonify, render_template
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager

import pandas as pd
import numpy as np
from io import BytesIO
from matplotlib import pyplot
import base64
import pickle
import shap
import requests
import os
import json
import orjson
import gc
import copy

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
        if shap_values.values[:, i][0][0]==0:
            signs.append(0)
        elif shap_values.values[:, i][0][0]>0:
            signs.append(1)
        else:
            signs.append(-1)

    df = pd.DataFrame({"feature_name": features, "importance": importances, "sign": signs})
    df.sort_values("importance", ascending=False, inplace=True)
    return df


def create_app(config={"TESTING": False, "TEMPLATES_AUTO_RELOAD": True}):
    api = Flask(__name__)

    with api.app_context():
        api.config.from_object(config)

        SECRET_KEY = os.urandom(32)
        api.config['SECRET_KEY'] = SECRET_KEY

        api.config["TEMPLATES_AUTO_RELOAD"] = True

        api.config["JWT_SECRET_KEY"] = SECRET_KEY
        jwt = JWTManager(api)

        api_initialized = False
        model = None
        X = None
        X_shap_global = None

        threshold = 0.16

        print("Initialisation débutée")
        if not os.path.exists("assets/model.pkl"):
            LOGIN = os.environ.get("parquet_get_login")
            assert LOGIN is not None
            PASSWORD = os.environ.get("parquet_get_password")
            assert PASSWORD is not None
            URL = os.environ.get("parquet_get_url")
            assert URL is not None
            ls = URL.rfind("/")
            url_model = URL[0:ls]+"/model.pkl"
            print("Téléchargement du fichier de modèle")
            response = requests.get(url = url_model, auth=(LOGIN,PASSWORD))
            with open("assets/model.pkl", "wb") as model_file:
                model_file.write(response.content)
            print("Fichier de modèle téléchargé")


        model = pickle.load(open("assets/model.pkl", "rb"))

        if not os.path.exists("assets/imputer.pkl"):
            LOGIN = os.environ.get("parquet_get_login")
            assert LOGIN is not None
            PASSWORD = os.environ.get("parquet_get_password")
            assert PASSWORD is not None
            URL = os.environ.get("parquet_get_url")
            assert URL is not None
            ls = URL.rfind("/")
            url_model = URL[0:ls]+"/imputer.pkl"
            print("Téléchargement du fichier imputer")
            response = requests.get(url = url_model, auth=(LOGIN,PASSWORD))
            with open("assets/imputer.pkl", "wb") as imputer_file:
                imputer_file.write(response.content)
            print("Fichier imputer téléchargé")

        imputer = pickle.load(open("assets/imputer.pkl", "rb"))

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
            print("Fichier parquet téléchargé")


        X = pd.read_parquet("assets/df_application_test.parquet")
        arr_sk_id_curr = X["SK_ID_CURR"].to_list()

        print("Création des données globales pour SHAP")
        X_shap_global = X.sample(200)
        del X

        explainer = shap.TreeExplainer(model)
        print("Calcul des valeurs Shapley globales")
        shap_values_global = explainer(X_shap_global.drop(columns=["SK_ID_CURR"]))
        print("Génération des données pour l'explication globale du modèle")
        shap_values_global_copy = copy.deepcopy(shap_values_global)
        shap_values_global_copy.values = shap_values_global_copy.values[:, :, 1]
        shap_values_global_copy.base_values = shap_values_global_copy.base_values[:, 1]

        columns_description = {}
        with open("assets/cols_description.json", "r") as f:
            columns_description = json.load(f)

        api_initialized = True
        print("Initialisation terminée")

        gc.collect()

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
        sk_id_curr = X.iloc[ind:ind+1]["SK_ID_CURR"][0]
        X_sample = X_shap_global.copy(deep=True)
        if X_sample.loc[X_sample["SK_ID_CURR"] == sk_id_curr].shape[0] == 0:
            X_sample = pd.concat([X.loc[X["SK_ID_CURR"] == sk_id_curr], X_sample])

        X_sample.reset_index(drop=True, inplace=True)
        idx = X_sample.loc[X_sample["SK_ID_CURR"] == sk_id_curr].index[0]
        X_sample.drop(columns=["SK_ID_CURR"], inplace=True)
        shap_values = explainer(imputer.transform(X_sample))

        vals = np.abs(shap_values.values[0]).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(X_sample.columns, vals)),
            columns=['col_name', 'feature_importance_vals']
        )
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)


        if show_plot:
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[1],
                shap_values[idx].values[:, 1],
                feature_names=X_sample.columns,
                max_display=max_display,
                show=True
            )

        if return_base64:
            global_exp_img = f"assets/global_explanation_{max_display}.png"
            if not os.path.exists(global_exp_img):
                pyplot.clf()
                shap.summary_plot(shap_values_global_copy, max_display=max_display, show=False)
                pyplot.savefig(global_exp_img, format='png')
            with open(global_exp_img, "rb") as global_img_f:
                global_expl_contents = base64.encodebytes(global_img_f.read()).decode('utf-8')



            pyplot.clf()
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[1],
                shap_values[idx].values[:, 1],
                feature_names=X_sample.columns,
                max_display=max_display+1,
                show=False
            )

            image = BytesIO()
            pyplot.savefig(image, format='png',bbox_inches='tight')
            ret = {
                "success": True,
                "image": base64.encodebytes(image.getvalue()).decode('utf-8'),
                "global_image": global_expl_contents,
                "features_importances": []
            }

            # let s continue with the features distributions
            df_feat_impo = get_df_feature_importances_shap_values(shap_values, X.drop(columns="SK_ID_CURR").columns)
            for i in range(0, max_display):
                line = df_feat_impo.iloc[i]
                fname = f"""assets/hist_json/{line["feature_name"].replace("/", "_")}.json"""
                with open(fname) as json_f:
                    decoded = json.load(json_f)
                    feature_name = str(line["feature_name"])
                    if feature_name in columns_description.keys():
                        feature_description = columns_description[feature_name]
                    else:
                        feature_description = ""
                    ret["features_importances"].append({
                        "name": feature_name,
                        "value": float(X.loc[ind, line["feature_name"]]),
                        "contribution": float(line["importance"] * line["sign"]),
                        "hist_json_y": list(float(y) for y in decoded[0]),
                        "hist_json_x": list(float(x) for x in decoded[1]),
                        "description": feature_description
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
    def health():
        return f"api_initialized={api_initialized}"


    @api.route("/login", methods=["GET"])
    def login_form():
        return render_template('login.html')

    # Create a route to authenticate your users and return JWTs. The
    # create_access_token() function is used to actually generate the JWT.
    @api.route("/login", methods=["POST"])
    def login_check():
        username = request.json.get("username", None)
        password = request.json.get("password", None)
        if username != "test" or password != "test":
            return {
                "success": False,
                "message": "Mauvais nom d'utilisateur ou mot de passe"
            }
    
        access_token = create_access_token(identity=username)
        return {
            "success": True,
            "access_token": access_token
        }

    
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
        ret = get_prediction(X, sk_id_curr, max_display)
        del X
        gc.collect()
        return ret


    @api.route('/predict2/<sk_id_curr>', methods = ['POST'])
    @jwt_required()
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
        ret = get_prediction(X, sk_id_curr, max_display)
        del X
        gc.collect()
        return ret

    return api




if __name__ == "__main__":
    api = create_app({"TESTING": False})
    api.run(host='0.0.0.0', port=8000)
