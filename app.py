import flask
from flask import Flask, request, render_template
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Load the trained model and scaler ---
try:
    from tensorflow.keras.metrics import MeanSquaredError
    model = tf.keras.models.load_model('shelf_life_model.h5', custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# --- Define the features your model expects ---
# IMPORTANT: Replace these with the actual column names from your dataset
# EXCEPT for the target variable ('Shelf_Life_Days')
FEATURE_COLUMNS = [
    'Storage_Temperature_C',
    'Initial_pH',
    'Final_pH',
    'Titratable_Acidity_g/100mL',
    'Microbial_Count_CFU/mL'
]

# --- Routes ---

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        print("Error: Model or scaler not loaded in predict route.")
        return "Error: Model or scaler not loaded. Please check server logs."

    try:
        print("--- Starting predict route ---")
        data = request.form.to_dict()
        print(f"Received data: {data}")

        input_data_dict = {col: None for col in FEATURE_COLUMNS}

        for key, value in data.items():
            if key in FEATURE_COLUMNS:
                try:
                    input_data_dict[key] = float(value)
                except ValueError:
                    print(f"Error: Invalid input for '{key}': {value}")
                    return f"Error: Invalid input for '{key}'. Please enter a number."

        print("Creating pandas DataFrame...")
        input_data = pd.DataFrame([input_data_dict])
        print(f"Input DataFrame created: {input_data}")

        print("Ensuring column order...")
        input_data = input_data[FEATURE_COLUMNS]
        print(f"Input DataFrame reordered: {input_data}")

        if input_data.isnull().values.any():
            print("Error: Missing values detected in input.")
            return "Error: Missing values detected in input. Please provide all feature values."

        print("Scaling input data...")
        input_data_scaled = scaler.transform(input_data)
        print(f"Input data scaled. Shape: {input_data_scaled.shape}")

        print("Making prediction...")
        prediction = model.predict(input_data_scaled)
        print(f"Raw prediction result: {prediction}")

        predicted_shelf_life = round(prediction[0][0], 2)
        print(f"Predicted shelf life: {predicted_shelf_life}")

        print("Rendering result template...")
        return render_template('result.html', prediction=predicted_shelf_life)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return f"An error occurred during prediction: {e}"

# --- Run the Flask application ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use Render's port or default to 10000
    app.run(debug=False, host='0.0.0.0', port=port)
