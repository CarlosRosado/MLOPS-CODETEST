import requests

SKLEARN_ENDPOINT = 'http://localhost:3000/sklearn'
PYTORCH_ENDPOINT = 'http://localhost:3000/pytorch'
ASTROMECH_ENDPOINT = 'http://localhost:3000/astromech'

class TestStationService:

    def test_successful_single_sklearn_response(self):
        crystal_data = [[0.92, 0.12, 0.31, 0.09]]
        input_data = {
            'crystalData': crystal_data
        }

        response = requests.post(SKLEARN_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_successful_batch_sklearn_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.7312, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = requests.post(SKLEARN_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_fail_single_sklearn_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09, 0.012],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = requests.post(SKLEARN_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_fail_batch_sklearn_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = requests.post(SKLEARN_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_successful_single_pytorch_response(self):
        crystal_data = [[0.92, 0.12, 0.31, 0.09]]
        input_data = {
            'crystalData': crystal_data
        }

        response = requests.post(PYTORCH_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_successful_batch_pytorch_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.7312, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data
        }
        print("soy el pytorch------------------------")
        response = requests.post(PYTORCH_ENDPOINT, json=input_data)
        print(f"response: {response}")
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_fail_single_pytorch_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09, 0.012],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = requests.post(PYTORCH_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_fail_batch_pytorch_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data
        }

        response = requests.post(PYTORCH_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_successful_single_astromech_response(self):
        crystal_data = [[0.92, 0.12, 0.31, 0.09]]
        input_data = {
            'crystalData': crystal_data,
            'model': 'sklearn'
        }

        response = requests.post(ASTROMECH_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_successful_batch_astromech_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.7312, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data,
            'model': 'sklearn'
        }

        response = requests.post(ASTROMECH_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 200
        assert 'prediction' in body
        assert 'scores' in body
        assert isinstance(body['prediction'][0], str)
        assert len(body['scores']) == len(crystal_data)

    def test_fail_single_astromech_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09, 0.012],
            ]
        input_data = {
            'crystalData': crystal_data,
            'model': 'sklearn'
        }

        response = requests.post(ASTROMECH_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_fail_batch_astromech_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data,
            'model': 'sklearn'
        }

        response = requests.post(ASTROMECH_ENDPOINT, json=input_data)
        status_code = response.status_code
        body = response.json()
        assert status_code == 422

    def test_fail_model_astromech_response(self):
        crystal_data = [
                [0.92, 0.12, 0.31, 0.09],
                [0.31, 0.112, 0.311, 0.09],
                [0.9212, 0.1112, 0.931, 0.409],
                [0.43921, 0.1222, 0.22, 0.0911],
                [0.93, 0.122, 0.311, 0.12],
                [0.64, 0.51, 0.92312, 0.329],
                [0.32, 0.32, 0.43],
                [0.90, 0.124, 0.131, 0.12],
            ]
        input_data = {
            'crystalData': crystal_data,
            'model': 'notexistent'
        }

        response = requests.post(ASTROMECH_ENDPOINT, json=input_data)
        
        status_code = response.status_code
        body = response.json()
        assert status_code == 400