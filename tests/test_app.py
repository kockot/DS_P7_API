import pytest

from app import create_app
import json
import base64
import cv2
import numpy as np

SECURITY_TOKEN = os.environ.get("security_token")
assert SECURITY_TOKEN is not None

@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def images_are_the_same(file1_name, img2_str):
    a = cv2.imread(file1_name)

    nparr = np.fromstring(img2_str, np.uint8)
    b = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

    difference = cv2.subtract(a, b)    
    return not np.any(difference)

    
def test_should_status_code_ok(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.data == b"api_initialized=True"


def test_negative_sk_id_curr(client):
    response = client.get("/predict/-999999", headers={"Authorization": SECURITY_TOKEN})
    data = json.loads(response.data)
    assert data["success"] == False
    assert data["message"] == "SK_ID_CURR n'est pas un entier naturel"


def test_float_sk_id_curr(client):
    response = client.get("/predict/125.5", headers={"Authorization": SECURITY_TOKEN})
    data = json.loads(response.data)
    assert data["success"] == False
    assert data["message"] == "SK_ID_CURR n'est pas un entier naturel"

    
def test_non_numeric_sk_id_curr(client):
    response = client.get("/predict/ABCDE", headers={"Authorization": SECURITY_TOKEN})
    data = json.loads(response.data)
    assert data["success"] == False
    assert data["message"] == "SK_ID_CURR n'est pas un entier naturel"

def test_application_credit_accepted(client):
    response = client.get("/predict/100038?max_display=50", headers={"Authorization": SECURITY_TOKEN})
    data = json.loads(response.data)
    #print(data)
    assert data["success"] == True
    assert data["conclusion"] == 1
    assert images_are_the_same("tests/images/100038.png", base64.decodebytes(bytes(data["image"], 'utf-8')))

    
    #f = open("tests/images/100038.png", "wb")
    #f.write(base64.decodebytes(bytes(data["image"], 'utf-8')))
    #f.close()
    

def test_application_credit_refused(client):
    response = client.get("/predict/456122?max_display=200", headers={"Authorization": SECURITY_TOKEN})
    data = json.loads(response.data)
    #print(data)
    assert data["success"] == True
    assert data["conclusion"] == 0
    assert images_are_the_same("tests/images/456122.png", base64.decodebytes(bytes(data["image"], 'utf-8')))

    
    #f = open("tests/images/456122.png", "wb")
    #f.write(base64.decodebytes(bytes(data["image"], 'utf-8')))
    #f.close()

def test_security_token_not_provided(client):
    response = client.get("/predict/456122?max_display=200")
    data = json.loads(response.data)
    #print(data)
    assert data["success"] == False
    assert data["message"] == "Jeton d'authentification non fourni"

def test_wrong_security_token(client):
    response = client.get("/predict/456122?max_display=200", headers={"Authorization": "nordine"})
    data = json.loads(response.data)
    #print(data)
    assert data["success"] == False
    assert data["message"] == "Echec de l'authentification du jeton"
