import joblib
import pandas as pd
import sys

# Load the saved model
try:
    model = joblib.load('shelf_life_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: shelf_life_model.pkl not found.")
    sys.exit(1)

def predict_shelf_life(input_data):
    """
    Predicts shelf life based on input data.

    Args:
        input_data (dict): A dictionary where keys are feature names
                           and values are the corresponding input values.

    Returns:
        float: The predicted shelf life.
    """
    # Convert input data to a pandas DataFrame
    # Ensure the order of columns matches the training data
    # (You'll need to adjust this based on your actual feature names)
    try:
        input_df = pd.DataFrame([input_data])
        # Make sure column order matches the features used for training
        # Get feature names from the training data (assuming you have X_train available)
        # If not, you'll need to store your feature names separately
        # For simplicity, assuming a fixed order based on the original features list
        # YOU WILL NEED TO VERIFY AND ADJUST THIS PART based on your features
        feature_names = ['feature1', 'feature2', 'feature3'] # Replace with your actual feature names
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]
        return prediction
    except KeyError as e:
        print(f"Error: Missing feature in input data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# Example usage (for testing the script directly)
if __name__ == "__main__":
    # Replace with sample input data based on your features
    sample_input = {'feature1': 10, 'feature2': 25, 'feature3': 1.5}
    predicted_value = predict_shelf_life(sample_input)

    if predicted_value is not None:
        print(f"Predicted shelf life: {predicted_value:.2f}")
