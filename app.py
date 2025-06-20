from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import sys
import os

# Define the default values for optional inputs
# YOU MUST REPLACE THE PLACEHOLDER VALUES BELOW with the actual mean values
# from your training data for Titratable Acidity and Microbial Count.
# Calculate these means after loading and cleaning your dataset.
DEFAULT_INITIAL_PH = 6.5
DEFAULT_TITRATABLE_ACIDITY = 0.15  # <== REPLACE WITH ACTUAL MEAN FROM YOUR DATA
DEFAULT_MICROBIAL_COUNT = 1000.0  # <== REPLACE WITH ACTUAL MEAN FROM YOUR DATA

# Load the saved models
try:
    model_final_ph = joblib.load('final_ph_model.pkl')
    print("Final_pH model loaded successfully.")
    model_shelf_life = joblib.load('shelf_life_model.pkl')
    print("Shelf_Life_Days model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model file: {e}. Make sure both 'final_ph_model.pkl' and 'shelf_life_model.pkl' are present.")
    sys.exit(1) # Exit if model files are not found

# Define the list of features the models expect in the correct order
# Ensure these match the order used during model training
MODEL_FEATURES_PH = ['Storage_Temperature_C', 'Initial_pH', 'Titratable_Acidity_g/100mL', 'Microbial_Count_CFU/mL']
MODEL_FEATURES_SHELF_LIFE = ['Storage_Temperature_C', 'Initial_pH', 'Titratable_Acidity_g/100mL', 'Microbial_Count_CFU/mL', 'Final_pH']


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
        # print("Received data:", data) # Uncomment for debugging

        # --- Process User Input ---
        # Apply default values based on dropdown selections
        storage_temp = 5.0 if data.get('storage_temp_option') == 'refrigerated' else \
                       25.0 if data.get('storage_temp_option') == 'room_temp' else \
                       float(data.get('storage_temp_specify', 5.0)) # Default to 5 if specify is chosen but no value provided


        initial_ph = DEFAULT_INITIAL_PH if data.get('initial_ph_option') == 'dont_know' else \
                     float(data.get('initial_ph_specify', DEFAULT_INITIAL_PH)) # Default to default if specify is chosen but no value provided


        titratable_acidity = DEFAULT_TITRATABLE_ACIDITY if data.get('titratable_acidity_option') == 'dont_know' else \
                            float(data.get('titratable_acidity_specify', DEFAULT_TITRATABLE_ACIDITY)) # Default to default if specify is chosen but no value provided


        microbial_count = DEFAULT_MICROBIAL_COUNT if data.get('microbial_count_option') == 'dont_know' else \
                         float(data.get('microbial_count_specify', DEFAULT_MICROBIAL_COUNT)) # Default to default if specify is chosen but no value provided


        # --- Create Input for Final_pH Model ---
        # Use the processed input data to predict Final_pH
        input_data_for_final_ph = {
            'Storage_Temperature_C': storage_temp,
            'Initial_pH': initial_ph,
            'Titratable_Acidity_g/100mL': titratable_acidity,
            'Microbial_Count_CFU/mL': microbial_count
        }

        try:
            input_df_ph = pd.DataFrame([input_data_for_final_ph])
            # Ensure column order matches the training data for Final_pH model
            input_df_ph = input_df_ph[MODEL_FEATURES_PH]

            predicted_final_ph = model_final_ph.predict(input_df_ph)[0]

            # --- Create Input for Shelf_Life_Days Model ---
            # Use the original processed input data PLUS the predicted Final_pH
            input_data_for_shelf_life = input_data_for_final_ph.copy() # Start with original inputs
            input_data_for_shelf_life['Final_pH'] = predicted_final_ph # Add the predicted Final_pH

            input_df_sl = pd.DataFrame([input_data_for_shelf_life])
            # Ensure column order matches the training data for Shelf_Life_Days model
            input_df_sl = input_df_sl[MODEL_FEATURES_SHELF_LIFE]


            predicted_shelf_life = model_shelf_life.predict(input_df_sl)[0]

            # Return both predictions
            return jsonify({
                'predicted_final_ph': predicted_final_ph,
                'predicted_shelf_life': predicted_shelf_life
            })

        except KeyError as e:
            print(f"Error processing input data: Missing expected key {e}")
            return jsonify({'error': f'Invalid input data: Missing expected field {e}'}), 400
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return jsonify({'error': 'An error occurred during prediction'}), 500

# This block is for running the Flask app locally (optional for Render)
if __name__ == '__main__':
    # Render will set the PORT environment variable
    port = int(os.environ.get('PORT', 5000)) # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True) # Set debug=True for local development
