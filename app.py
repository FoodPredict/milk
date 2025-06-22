from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import sys
import os

# Define the default values for optional inputs
# YOU MUST REPLACE THE PLACEHOLDER VALUES BELOW with the actual mean values
# from your training data for Titratable Acidity and Microbial Count.
# Calculate these means after loading and cleaning your dataset.
DEFAULT_INITIAL_PH = 5.5
DEFAULT_TITRATABLE_ACIDITY = 0.15  # <== REPLACE WITH ACTUAL MEAN FROM YOUR DATA
DEFAULT_MICROBIAL_COUNT = 10000.0  # <== REPLACE WITH ACTUAL MEAN FROM YOUR DATA
# You might also need default values for packaging if it's optional, but it seems required by your new model.

# Load the saved models
try:
    model_final_ph = joblib.load('final_ph_model.pkl')
    print("Final_pH model loaded successfully.")
    model_shelf_life = joblib.load('shelf_life_model.pkl')
    print("Shelf_Life_Days model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model file: {e}. Make sure both 'final_ph_model.pkl' and 'shelf_life_model.pkl' are present.")
    sys.exit(1) # Exit if model files are not found

# Define the list of features the models expect BEFORE one-hot encoding for PH model
# These should match the features_for_final_ph list from your notebook
INPUT_FEATURES_PH = ['Storage_Temperature_C', 'Initial_pH', 'Titratable_Acidity_g/100mL', 'Microbial_Count_CFU/mL', 'Packaging']

# The features for the shelf life model will be the encoded PH features + Final_pH
# We'll determine the exact list of features after encoding
# MODEL_FEATURES_SHELF_LIFE = ['Storage_Temperature_C', 'Initial_pH', 'Titratable_Acidity_g/100mL', 'Microbial_Count_CFU/mL', 'Final_pH'] # This needs to be updated


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
        # Note: It looks like the rest of the processing logic for data
        # is missing here. You need to extract all the required features
        # from the `data` dictionary received from the frontend.
        # Assuming the keys in the JSON data match the feature names:

        try:
            storage_temp = float(data.get('Storage_Temperature_C', 5.0 if data.get('storage_temp_option') == 'refrigerated' else 25.0)) # Adjust default based on storage_temp_option if it exists
            initial_ph = float(data.get('Initial_pH', DEFAULT_INITIAL_PH))
            titratable_acidity = float(data.get('Titratable_Acidity_g/100mL', DEFAULT_TITRATABLE_ACIDITY))
            microbial_count = float(data.get('Microbial_Count_CFU/mL', DEFAULT_MICROBIAL_COUNT))
            packaging = data.get('Packaging') # Assuming 'Packaging' is sent directly

            # Ensure packaging is provided, as it's a required feature now
            if packaging is None:
                 return jsonify({'error': 'Packaging information is required.'}), 400

        except ValueError as e:
            return jsonify({'error': f'Invalid input data: {e}'}), 400
        except KeyError as e:
             return jsonify({'error': f'Missing required input key: {e}'}), 400


        # Create a DataFrame for the single input instance
        # This DataFrame structure must match the one used for training BEFORE encoding
        input_df_ph = pd.DataFrame([{
            'Storage_Temperature_C': storage_temp,
            'Initial_pH': initial_ph,
            'Titratable_Acidity_g/100mL': titratable_acidity,
            'Microbial_Count_CFU/mL': microbial_count,
            'Packaging': packaging
        }])

        # Apply One-Hot Encoding to the input DataFrame
        # This must use the *same* columns list and drop_first setting as in training
        try:
            input_df_ph_encoded = pd.get_dummies(input_df_ph, columns=['Packaging'], drop_first=True)
        except Exception as e:
            # Catch potential errors during encoding (e.g., unseen packaging type)
             return jsonify({'error': f'Error during packaging encoding: {e}'}), 500


        # --- Predict Final pH ---
        # Ensure the encoded input DataFrame has the exact same columns as the X_train_ph
        # used during training for the Final_pH model.
        # You might need to store the columns from X_train_ph during training
        # and load them here to reindex the input_df_ph_encoded.
        # For demonstration, let's assume the columns are consistent.
        # A more robust approach involves saving and loading the training columns.

        # To handle potential missing dummy columns if the input doesn't have all packaging types
        # present in the training data, we can reindex and fill missing columns with 0.
        # This requires saving the list of columns from the training data's encoded features (X_final_ph).
        # Let's assume you have saved this list as 'final_ph_model_features.pkl' during training.
        # If you haven't, you should modify your training script to save it.

        try:
            # Load the list of features the Final_pH model was trained on
            # Make sure you save this list during training!
            # Example: joblib.dump(X_final_ph.columns.tolist(), 'final_ph_model_features.pkl')
            final_ph_model_features = joblib.load('final_ph_model_features.pkl')
            # Reindex the input DataFrame to match the training columns
            input_df_ph_aligned = input_df_ph_encoded.reindex(columns=final_ph_model_features, fill_value=0)
        except FileNotFoundError:
             return jsonify({'error': 'final_ph_model_features.pkl not found. Please train and save it.'}), 500
        except Exception as e:
             return jsonify({'error': f'Error aligning Final_pH input features: {e}'}), 500


        predicted_final_ph = model_final_ph.predict(input_df_ph_aligned)[0]


        # --- Prepare Data for Shelf Life Prediction ---
        # Combine the original encoded features with the predicted Final_pH
        input_df_shelf_life = input_df_ph_encoded.copy() # Start with the encoded features
        input_df_shelf_life['Final_pH'] = predicted_final_ph # Add the predicted Final_pH

        # Ensure the input DataFrame for the shelf life model has the exact same columns
        # as the X_train_sl used during training for the Shelf_Life_Days model.
        # Similar to the pH model, it's best to save and load the training columns here.
        # Let's assume you have saved this list as 'shelf_life_model_features.pkl' during training.
        # If you haven't, you should modify your training script to save it.

        try:
            # Load the list of features the Shelf_Life_Days model was trained on
            # Make sure you save this list during training!
            # Example: joblib.dump(X_shelf_life.columns.tolist(), 'shelf_life_model_features.pkl')
            shelf_life_model_features = joblib.load('shelf_life_model_features.pkl')
             # Reindex the input DataFrame to match the training columns
            input_df_shelf_life_aligned = input_df_shelf_life.reindex(columns=shelf_life_model_features, fill_value=0)
        except FileNotFoundError:
             return jsonify({'error': 'shelf_life_model_features.pkl not found. Please train and save it.'}), 500
        except Exception as e:
             return jsonify({'error': f'Error aligning Shelf_Life_Days input features: {e}'}), 500


        # --- Predict Shelf Life ---
        predicted_shelf_life = model_shelf_life.predict(input_df_shelf_life_aligned)[0]

        # Return the predictions
        return jsonify({
            'predicted_final_ph': float(predicted_final_ph), # Convert to float for JSON
            'predicted_shelf_life': float(predicted_shelf_life) # Convert to float for JSON
        })

    # If not POST, maybe return an error or method not allowed
    return jsonify({'error': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    # You might want to use a production-ready server like Gunicorn in deployment
    # For local testing:
    app.run(debug=True, host='0.0.0.0', port=5000)
Use code with caution
Rate this answer
