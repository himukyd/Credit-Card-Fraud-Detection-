import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Absolute Paths (use raw string to avoid issues with spaces) ---
MODEL_PATH = r"/Users/himanshuyadav/Desktop/IIT 3rd sem /DSAI/Project /Credit-Card-Fraud-Detection-/models/fraud_detection_model.pkl"
SCALER_PATH = r"/Users/himanshuyadav/Desktop/IIT 3rd sem /DSAI/Project /Credit-Card-Fraud-Detection-/models/scaler.pkl"

# --- Load model and scaler safely ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading model or scaler:", e)
