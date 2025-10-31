from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('../models/fraud_detection_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values
        features = [float(x) for x in request.form.values()]
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]
        result = "🚨 Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
