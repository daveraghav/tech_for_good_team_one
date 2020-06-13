from flask import Flask, request, jsonify
import joblib
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

model = joblib.load('model.h5')

def return_prediction(model, sample_json):
    
    input_data = [
        int(sample_json['Age']),
        int(sample_json['gender'].lower() in ['m', 'male']),
        int(sample_json['gender'].lower() not in ['m', 'male', 'f', 'female']),
        int(sample_json['treatment'].lower() in ['y', 'yes']),
        int(sample_json['family_history'].lower() in ['y', 'yes']),
        int(sample_json['obs_consequence'].lower() in ['y', 'yes']),
        int(sample_json['leave_Very_difficult'].lower() in ['y', 'yes']),
        int(sample_json['mental_vs_physical'].lower() in ['n', 'no']),
    ]
    
    
    prediction = model.predict(np.array(input_data).reshape((1,-1)))
    
    prediction_map = {
        0:'Never',
        1:'Rarely',
        2:'Sometimes',
        3:'Often'
    }    

    return prediction_map[prediction[0]]
    
    

@app.route("/")
def index():
    return "<h1>Flask app is running</h1>"

@app.route('/api/predict', methods=['POST'])
def flower_prediction():
    content = request.json
    results = return_prediction(model, content)
    return jsonify(results)

if __name__=='__main__':
    app.run()