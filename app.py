import flask
from flask import Flask, request, render_template
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# --- Load the trained model and scaler ---
try:
    # Add this import if you haven't already in app.py
    from tensorflow.keras.metrics import MeanSquaredError

    # Load the model, specifying MSE as a custom object
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
    """Handles the prediction request from the form."""
    if model is None or scaler is None:
        return "Error: Model or scaler not loaded. Please check server logs."

    try:
        # Get data from the form
        data = request.form.to_dict()

        # Create a dictionary with all expected feature columns, initialized to None or a default
        # This helps ensure all columns are present, even if not submitted in the form (though your form should include all)
        input_data_dict = {col: None for col in FEATURE_COLUMNS}

        # Update the dictionary with the actual submitted data
        for key, value in data.items():
             if key in FEATURE_COLUMNS:
                 try:
                     # Attempt to convert input values to float
                     input_data_dict[key] = float(value)
                 except ValueError:
                     # Handle cases where input is not a valid number
                     return f"Error: Invalid input for '{key}'. Please enter a number."


        # Convert the dictionary to a pandas DataFrame
        # The order of columns in the DataFrame must match the order used during training
        # Creating DataFrame from a list of dicts is a good way to handle this
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
        # prediction[0][0] extracts the single prediction value from the numpy array
        predicted_shelf_life = round(prediction[0][0], 2) # Round to 2 decimal places

        return render_template('result.html', prediction=predicted_shelf_life)

    except Exception as e:
        # Catch any other unexpected errors during prediction
        return f"An error occurred during prediction: {e}"

# --- Run the Flask application ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the server accessible externally for deployment
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0')
