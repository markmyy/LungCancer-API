from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("lungcancer.pkl")

@app.route('/api/lungcancer', methods=['POST'])
def lungcancer():
    gender = int(request.form.get('gender')) 
    age = int(request.form.get('age')) 
    smoking = int(request.form.get('smoking')) 
    chro = int(request.form.get('chro')) 
    fatigue = int(request.form.get('fatigue')) 
    allegery = int(request.form.get('allegery')) 
    coughing = int(request.form.get('coughing')) 

    # Prepare the input for the model
    x = np.array([[gender, age, smoking, chro, fatigue, allegery, coughing]])

    # Predict using the model
    prediction = model.predict(x)
    
    # Check the prediction result
    if int(prediction[0]) == 0:
        return {'Prediction': 'เป็นมะเร็ง'}
    else:
        return {'Prediction': 'ไม่เป็นมะเร็ง'}

# Run the server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
