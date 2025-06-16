import flask
from flask import Flask, request, render_template
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np

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

# ... (rest of your Flask code)

    @app.route('/predict', methods=['POST'])
    def predict():
        if model is None or scaler is None:
            print("Error: Model or scaler not loaded in predict route.")
            return "Error: Model or scaler not loaded. Please check server logs."

        try:
            print("--- Starting predict route ---") # Added
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

            print("Creating pandas DataFrame...") # Added
            input_data = pd.DataFrame([input_data_dict])
            print(f"Input DataFrame created: {input_data}")

            print("Ensuring column order...") # Added
            input_data = input_data[FEATURE_COLUMNS]
            print(f"Input DataFrame reordered: {input_data}")

            if input_data.isnull().values.any():
                 print("Error: Missing values detected in input.")
                 return "Error: Missing values detected in input. Please provide all feature values."

            print("Scaling input data...") # Added
            input_data_scaled = scaler.transform(input_data)
            print(f"Input data scaled. Shape: {input_data_scaled.shape}")

            print("Making prediction...") # Added
            prediction = model.predict(input_data_scaled)
            print(f"Raw prediction result: {prediction}")

            predicted_shelf_life = round(prediction[0][0], 2)
            print(f"Predicted shelf life: {predicted_shelf_life}")

            print("Rendering result template...") # Added
            return render_template('result.html', prediction=predicted_shelf_life)

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return f"An error occurred during prediction: {e}"
       
        # Convert the dictionary to a pandas DataFrame
        input_data = pd.DataFrame([input_data_dict])

        # Ensure columns are in the correct order
        input_data = input_data[FEATURE_COLUMNS]

        # Handle potential missing values or errors before scaling
        if input_data.isnull().values.any():
            return "Error: Missing values detected in input. Please provide all feature values."

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)

        # Return the prediction result
        predicted_shelf_life = round(prediction[0][0], 2) # Round to 2 decimal places

        return render_template('result.html', prediction=predicted_shelf_life)

        except Exception as e:
        return f"An error occurred during prediction: {e}"

# --- Run the Flask application ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
