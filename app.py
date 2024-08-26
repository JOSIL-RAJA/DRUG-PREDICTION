from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
with open('drug_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
le_sex = LabelEncoder()
le_sex.fit(['F', 'M'])

le_bp = LabelEncoder()
le_bp.fit(['LOW', 'NORMAL', 'HIGH'])

le_chol = LabelEncoder()
le_chol.fit(['NORMAL', 'HIGH'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    age = int(request.form['age'])
    sex = request.form['sex']
    bp = request.form['bp']
    cholesterol = request.form['cholesterol']
    na_to_k = float(request.form['na_to_k'])

    # Encode categorical variables
    sex_encoded = le_sex.transform([sex])[0]
    bp_encoded = le_bp.transform([bp])[0]
    cholesterol_encoded = le_chol.transform([cholesterol])[0]

    # Predict the drug type
    features = np.array([[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]])
    prediction = model.predict(features)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)
