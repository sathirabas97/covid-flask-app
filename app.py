import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import src.covid_model as cvd

import json

df = pd.read_excel('dataset.xlsx')
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return '''<h1>Welcome to Covid Prediction Model</h1>'''


@app.route('/health_check')
def health_check():
    return "Health check: OK"


@app.route('/api/run_the_model')
def run_the_model():
    cvd.run_the_model()

    return jsonify({"msg": f"model was trained and saved"})


@app.route('/api/predict', methods=['post'])
def predict():
    Age = request.form.get('Age', type=int, default='')
    heart_beat = request.form.get('heart_beat', type=int, default='')
    Gender = request.form.get('Gender', type=int, default='')
    data = [Age, heart_beat, Gender]
    prediction = cvd.predict(data)
    print(prediction)
    return jsonify({"Number_of_steps": f"{prediction[0][0]}", "Sleep": f"{prediction[0][1]}"})


if __name__ == "__main__":
    app.run(threaded=True, host="0.0.0.0", port=5003)
