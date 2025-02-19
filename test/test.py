from fastapi.testclient import TestClient
from loguru import logger
from main import app
import json

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "This is a topic classification model for complaints"}
    logger.info("Test read root passed")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "OK"}
    logger.info("Test health check passed")
def test_predict_endpoint(payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    print("Result: ",response.json())

if __name__ == "__main__":

    with open("test/testdata.json", "r") as f:
        data = json.load(f)
    print("Total data: ",len(data))
    print("--------------------------------")
    test_read_root()
    test_health_check()
    print("--------------------------------")
    for i in range(len(data)):
        test_predict_endpoint(data[i])


