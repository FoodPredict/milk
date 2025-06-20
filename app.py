from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import sys
import os

# Assuming your prediction script is in the same directory
# If not, adjust the import path
try:
    # Load the saved model
    model = joblib.load('shelf_life_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: shelf_life_model.pkl not found.")
    sys.exit(1) # Exit if the model file is not found

# --- Function to predict shelf life (adapted from your script) ---
def predict_shelf_life(input_data):
    """
    Predicts shelf life based on input data.

    Args:
        input_data (dict): A dictionary where keys are feature names
                           and values are the corresponding input values.

    Returns:
        float: The predicted shelf life.
    """
    try:
        input_df = pd.DataFrame([input_data])
        # !!! IMPORTANT: Replace these with your actual feature names
        # Ensure the order matches the training data
        feature_names = ['feature1', 'feature2', 'feature3'] # <== ADJUST THIS
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]
        return prediction
    except KeyError as e:
        print(f"Error: Missing feature in input data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# --- Flask Web Application ---
app = Flask(__name__)

# Route to serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json() # Get JSON data from the request

        # Call the predict_shelf_life function with the received data
        prediction = predict_shelf_life(data)

        if prediction is not None:
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Could not process prediction'}), 400 # Bad Request

# This block is for running the Flask app locally (optional for Render)
if __name__ == '__main__':
    # Render will set the PORT environment variable
    port = int(os.environ.get('PORT', 5000)) # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
