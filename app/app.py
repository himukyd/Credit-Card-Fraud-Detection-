from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Absolute paths (update if needed)
MODEL_PATH = "/Users/himanshuyadav/Desktop/IIT 3rd sem /DSAI/Project /Credit-Card-Fraud-Detection-/models/fraud_detection_model.pkl"
SCALER_PATH = "/Users/himanshuyadav/Desktop/IIT 3rd sem /DSAI/Project /Credit-Card-Fraud-Detection-/models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print("❌ Error loading model or scaler:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [float(x) for x in request.form.values()]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        
        result = "🚨 Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction"
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('result.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
