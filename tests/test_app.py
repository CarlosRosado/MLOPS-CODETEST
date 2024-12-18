import pytest
import sys, os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_sklearn_endpoint(client):
    response = client.post('/sklearn', json={"crystalData": [[0.92, 0.12, 0.31, 0.09]]})
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert 'scores' in data

def test_pytorch_endpoint(client):
    response = client.post('/pytorch', json={"crystalData": [[0.92, 0.12, 0.31, 0.09]]})
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert 'scores' in data

def test_astromech_endpoint_sklearn(client):
    response = client.post('/astromech', json={"crystalData": [[0.92, 0.12, 0.31, 0.09]], "model": "sklearn"})
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert 'scores' in data

def test_astromech_endpoint_pytorch(client):
    response = client.post('/astromech', json={"crystalData": [[0.92, 0.12, 0.31, 0.09]], "model": "pytorch"})
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert 'scores' in data

def test_invalid_input_sklearn(client):
    response = client.post('/sklearn', json={"invalidData": [[0.92, 0.12, 0.31, 0.09]]})
    assert response.status_code == 422

def test_invalid_input_pytorch(client):
    response = client.post('/pytorch', json={"invalidData": [[0.92, 0.12, 0.31, 0.09]]})
    assert response.status_code == 422

def test_invalid_input_astromech(client):
    response = client.post('/astromech', json={"invalidData": [[0.92, 0.12, 0.31, 0.09]], "model": "pytorch"})
    assert response.status_code == 422

def test_invalid_model_astromech(client):
    response = client.post('/astromech', json={"crystalData": [[0.92, 0.12, 0.31, 0.09]], "model": "invalid"})
    assert response.status_code == 400