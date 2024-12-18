from flask import Flask, request, jsonify
import joblib
import torch
import numpy as np
import os,sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.pytorch_classifier import PytorchClassifier
from models.sklearn_classifier import SklearnClassifier

from utils import load_labels, validate_input, format_response

app = Flask(__name__)

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
sklearn_model_path = os.path.join(BASE_DIR, 'models', 'sklearn.model')
pytorch_model_path = os.path.join(BASE_DIR, 'models', 'pytorch.model')

sklearn_model = SklearnClassifier(sklearn_model_path)
pytorch_model = PytorchClassifier(pytorch_model_path)

# Load labels
labels = load_labels(os.path.join(BASE_DIR, 'models', 'output_labels.txt'))


@app.route('/sklearn', methods=['POST'])
def sklearn_endpoint():
    """
    Endpoint for making predictions using the Scikit-Learn model.

    Expects a JSON payload with the key 'crystalData' containing a list of samples.

    Returns:
        JSON response with the prediction and scores for each label.
    """
    data = request.get_json()

    if not validate_input(data):
        return jsonify({"error": "Invalid input data"}), 422
    try:
        predictions = sklearn_model.predict(data['crystalData'])
        response = format_response(predictions, labels)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/pytorch', methods=['POST'])
def pytorch_endpoint():
    """
    Endpoint for making predictions using the PyTorch model.

    Expects a JSON payload with the key 'crystalData' containing a list of samples.

    Returns:
        JSON response with the prediction and scores for each label.
    """
    data = request.get_json()
    if not validate_input(data):
        return jsonify({"error": "Invalid input data"}), 422
    try:
        predictions = pytorch_model.predict(data['crystalData'])
        response = format_response(predictions, labels)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/astromech', methods=['POST'])
def astromech_endpoint():
    """
    Endpoint for making predictions using either the Scikit-Learn or PyTorch model.

    Expects a JSON payload with the key 'crystalData' containing a list of samples,
    and a key 'model' specifying either 'sklearn' or 'pytorch'.

    Returns:
        JSON response with the prediction and scores for each label.
    """
    data = request.get_json()
    model_type = data['model']
    if model_type not in ['sklearn', 'pytorch']:
        return jsonify({"error": "Invalid model type"}), 400
    if not validate_input(data):
        return jsonify({"error": "Invalid input data"}), 422
    try:
        if model_type == 'sklearn':
            predictions = sklearn_model.predict(data['crystalData'])
        else:
            predictions = pytorch_model.predict(data['crystalData'])
        response = format_response(predictions, labels)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000)