import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder
from sklearn.preprocessing import StandardScaler # Ensure StandardScaler is imported

app = Flask(__name__)

# Define the base directory for saved files (models, scaler)
# We will refit encoders on the fly
base_dir = os.path.dirname(os.path.abspath(__file__)) # Assumes app.py is in the same directory as the pkl files

# Define paths to the saved files
model_path = os.path.join(base_dir, 'car_price_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')
# We won't load label_encoders.pkl directly, we will refit

# Path to the CSV file containing original string values
# IMPORTANT: YOU MUST PLACE a master_df.csv file saved from the notebook
# *before* final numerical encoding, containing ORIGINAL STRING VALUES,
# in the same directory as app.py
original_data_path = os.path.join(base_dir, 'master_df.csv')

# Initialize all_feature_cols_expected_by_model here to ensure it's always defined
# Provide a default fallback list that matches the columns used during training
all_feature_cols_expected_by_model = ['car_name', 'year', 'horsepower', 'mileage', 'fuel_type', 'transmission', 'owner_type']


# Load the model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully.")
    # If scaler has feature names, update the global variable
    if hasattr(scaler, 'feature_names_in_'):
        all_feature_cols_expected_by_model = list(scaler.feature_names_in_)
        print(f"Scaler feature names found: {all_feature_cols_expected_by_model}")


except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    print("Please ensure 'car_price_model.pkl' and 'scaler.pkl' are in the same directory as app.py")
    model = None
    scaler = None
    # all_feature_cols_expected_by_model will retain its default value

# Refit label encoders on original string data
label_encoders = {}
# Define all categorical columns expected by your model for encoding
# IMPORTANT: These names must exactly match the column names in your ORIGINAL unencoded data
# AND must be a subset of all_feature_cols_expected_by_model
categorical_cols_for_refitting = ['car_name', 'fuel_type', 'transmission', 'owner_type'] # Add other original categorical cols if they exist and are needed for dropdowns

try:
    # Load the CSV that contains original string values for refitting encoders
    # Make sure this file is saved from the notebook *before* encoding
    original_df_unencoded = pd.read_csv(original_data_path)
    print(f"Original data loaded successfully from {original_data_path}.")

    # Ensure columns used for fitting exist in the loaded data
    missing_cols_in_csv = [col for col in categorical_cols_for_refitting if col not in original_df_unencoded.columns]
    if missing_cols_in_csv:
        print(f"Warning: Missing categorical columns in {original_data_path}: {missing_cols_in_csv}. Cannot refit encoders for these.")

    for col in categorical_cols_for_refitting:
        if col in original_df_unencoded.columns:
            le = LabelEncoder()
            # Fit on the original string values from the loaded CSV
            # Convert to string to handle potential NaNs and ensure correct dtype
            le.fit(original_df_unencoded[col].astype(str).dropna()) # Fit on non-null string values
            label_encoders[col] = le
            print(f"Encoder refitted for '{col}' with {len(le.classes_)} classes.")
        else:
             # If column was missing, skip refitting
             print(f"Skipping encoder refitting for '{col}' as it's not in the loaded data.")
             pass # Encoder won't be in label_encoders dict


    print("Label encoders refitted successfully.")

except FileNotFoundError:
    print(f"Error: Original data file not found at {original_data_path}. Cannot refit label encoders.")
    print("Please ensure 'master_df.csv' (saved from notebook *before* final numerical encoding) is in the same directory as app.py")
    # If original data is not found, label_encoders will remain empty or incomplete
    label_encoders = {} # Initialize as empty dict on error


# Define numeric limits for input validation (example values, adjust as needed)
# These values should ideally be based on the min/max or reasonable ranges from your training data
numeric_limits = {
    'year': (1990, 2025),
    'mileage': (0, 500000),
    'horsepower': (30, 1000)
}

# Define the list of graph filenames saved earlier - REMOVED 'graphs/' prefix
# Ensure these filenames exactly match the files you placed directly in the 'static' folder
graph_filenames = [
    'year_distribution.png',
    'year_boxplot.png',
    'horsepower_distribution.png',
    'horsepower_boxplot.png',
    'mileage_distribution.png',
    'mileage_boxplot.png',
    'price_distribution.png',
    'price_boxplot.png',
    'fuel_type_count.png',
    'transmission_count.png',
    'owner_type_count.png',
    'year_vs_price_scatterplot.png',
    'horsepower_vs_price_scatterplot.png',
    'mileage_vs_price_scatterplot.png',
    'fuel_type_vs_price_boxplot.png',
    'transmission_vs_price_boxplot.png',
    'owner_type_vs_price_boxplot.png',
    'correlation_heatmap.png'
]


@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    if model is None or scaler is None:
         return "Error: Model or Scaler could not be loaded. Check logs for details.", 500
    if not label_encoders: # Check if label_encoders dict is empty
        return "Error: Label encoders could not be loaded/refitted. Ensure original data CSV is correct and present.", 500


    # Prepare data for dropdowns in the HTML form - Pass the original string classes
    # Only include dropdown data for columns where encoders were successfully refitted
    dropdown_data = {key: list(enc.classes_) for key, enc in label_encoders.items() if key in label_encoders}


    # all_feature_cols_expected_by_model is a global variable


    return render_template('index.html',
                           dropdown_data=dropdown_data,
                           numeric_limits=numeric_limits,
                           # Pass the expected feature order if available
                           feature_columns_order=all_feature_cols_expected_by_model
                          )

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    # Access the global variable
    global all_feature_cols_expected_by_model

    if model is None or scaler is None or not label_encoders:
        return "Model components not loaded correctly. Cannot make prediction.", 500

    try:
        # Get data from the form
        # Use request.form.get() to handle potential missing keys gracefully (returns None if key not found)
        user_input_raw = {
            'car_name': request.form.get('car_name'),
            'year': request.form.get('year'),
            'horsepower': request.form.get('horsepower'),
            'mileage': request.form.get('mileage'),
            'fuel_type': request.form.get('fuel_type'),
            'transmission': request.form.get('transmission'),
            'owner_type': request.form.get('owner_type')
            # Add other fields from your HTML form here if applicable
        }


        # Prepare data for DataFrame, ensuring all expected columns are present
        # Use the global variable for expected order
        input_data_processed = {}
        for col in all_feature_cols_expected_by_model:
             input_data_processed[col] = user_input_raw.get(col, None) # Use .get with default None if form field is missing

        input_df = pd.DataFrame([input_data_processed]) # Create DataFrame with one row


        # Preprocess the input data - Categorical Encoding
        categorical_cols_to_encode = list(label_encoders.keys()) # Get columns for which we have encoders

        for col in categorical_cols_to_encode:
             # Ensure the column exists in input_df and the encoder exists for this column
             if col in input_df.columns and col in label_encoders:
                # Get the value, handle None/NaN from raw input
                input_val = input_df[col].iloc[0]
                # Convert to string, handle potential NaN/None from raw input and empty strings from form
                input_val_str = str(input_val).lower() if pd.notna(input_val) and input_val != '' else label_encoders[col].classes_[0]


                # Handle unseen values during transformation
                if input_val_str not in label_encoders[col].classes_:
                    print(f"Warning: Unknown {col} '{input_val_str}'. Encoding it as the first class.")
                    default_value = label_encoders[col].classes_[0]
                    input_df[col] = label_encoders[col].transform([default_value])[0] # Transform the default value
                else:
                    # Transform the known value
                    input_df[col] = label_encoders[col].transform([input_val_str])[0]
             else:
                 # If column or encoder is missing, set to a default encoded value or handle as error
                 print(f"Error: Encoder for '{col}' not available or column missing in input_df. Cannot encode input.")
                 # Set to a default encoded value (e.g., 0) or handle as an error
                 # Assuming 0 is a safe default for encoded categorical features that model can handle
                 input_df[col] = 0


        # Convert numeric columns - Handle potential empty strings or None from request.form.get
        # Use the global all_feature_cols_expected_by_model to determine numeric columns
        numeric_cols_to_convert = [col for col in all_feature_cols_expected_by_model if col not in categorical_cols_to_encode]


        for col in numeric_cols_to_convert:
            if col in input_df.columns and input_df[col].iloc[0] is not None and input_df[col].iloc[0] != '':
                 # Use pd.to_numeric to handle conversion, errors='coerce' will turn invalid values into NaN
                 input_df[col] = pd.to_numeric(input_df[col].iloc[0], errors='coerce')
            else:
                 input_df[col] = np.nan # Set to NaN if empty or None


        # Impute missing numeric values using scaler's mean (approximation of training data median)
        numeric_cols_for_imputation = numeric_cols_to_convert # Impute the same columns we tried to convert
        for col in numeric_cols_for_imputation:
             if col in input_df.columns and pd.isna(input_df[col].iloc[0]):
                 # Find the index of the column in the scaler's feature_names_in_
                 try:
                     # Ensure scaler.feature_names_in_ is available and check column names
                     if hasattr(scaler, 'feature_names_in_') and col in scaler.feature_names_in_:
                         col_index_in_scaler = list(scaler.feature_names_in_).index(col)
                         input_df[col] = scaler.mean_[col_index_in_scaler] # Impute with the mean from scaler
                     else:
                         print(f"Warning: Scaler feature names not available or column '{col}' not found. Cannot impute using scaler mean.")
                         input_df[col] = 0 # Fallback: Impute with 0 if scaler info is missing
                 except Exception as e:
                     print(f"Error during imputation for column '{col}': {e}")
                     input_df[col] = 0 # Fallback on error


        # Ensure input_df has the correct column order and names matching scaler.feature_names_in_
        # This is critical for scaler.transform and model.predict
        # Use the global variable for expected order
        try:
             # Reindex the input_df to ensure correct column order and presence
             # This is a safer way to ensure the input DataFrame matches the expected structure
             input_df_processed_final = input_df.reindex(columns=all_feature_cols_expected_by_model)


             # Need to handle potential dtypes after reindex, esp. for encoded categorical columns
             # and imputed numeric columns. Ensure all are numeric before scaling.
             # Convert all columns to numeric, coercing errors to NaN (should have been handled by imputation)
             for col in input_df_processed_final.columns:
                  input_df_processed_final[col] = pd.to_numeric(input_df_processed_final[col], errors='coerce')
                  # Re-impute if pd.to_numeric created new NaNs (unlikely if previous steps were robust)
                  if pd.isna(input_df_processed_final[col].iloc[0]):
                       print(f"Warning: NaN created after numeric conversion for '{col}'. Imputing again.")
                       if col in numeric_cols_for_imputation:
                           # Impute missing numeric column with scaler mean (or fallback)
                           try:
                                if hasattr(scaler, 'feature_names_in_') and col in scaler.feature_names_in_:
                                    col_index_in_scaler = list(scaler.feature_names_in_).index(col)
                                    input_df_processed_final[col] = scaler.mean_[col_index_in_scaler]
                                else:
                                     input_df_processed_final[col] = 0 # Fallback
                           except:
                                input_df_processed_final[col] = 0 # Fallback
                       elif col in categorical_cols_to_encode:
                            # Impute missing categorical column with a default encoded value (e_g_ 0)
                            input_df_processed_final[col] = 0 # Assuming 0 is a safe default
                       else:
                           # For any other unexpected missing column, set to 0 or NaN
                           input_df_processed_final[col] = 0 # Defaulting to 0


             # Apply the scaler to the prepared and ordered input data
             # Pass the DataFrame directly to scaler.transform and model.predict
             input_scaled_df = scaler.transform(input_df_processed_final) # scaler.transform returns a numpy array by default

        except Exception as e:
             print(f"Error preparing or scaling input data: {e}")
             # Handle preparation/scaling error - return an error message to the user
             return f"Error processing input data for prediction: {e}", 500


        # Make the prediction
        # Ensure the model is expecting the scaled data shape and format
        # Convert scaled numpy array back to DataFrame with feature names before prediction
        input_scaled_df_with_names = pd.DataFrame(input_scaled_df, columns=all_feature_cols_expected_by_model)

        prediction = model.predict(input_scaled_df_with_names)
        predicted_price = prediction[0]

        # Format the predicted price (e.g., to two decimal places, add currency symbol)
        formatted_price = f"Predicted Price: ₹{predicted_price:,.2f}" # Example formatting for Rupees

        # To display the prediction result back on the index.html,
        # we render the template again, passing the prediction result
        # along with the dropdown and numeric limits data
        dropdown_data_for_template = {key: list(enc.classes_) for key, enc in label_encoders.items() if key in label_encoders}

        # Re-render index.html with prediction result and potentially submitted form data to keep form populated
        # To repopulate the form, you need to pass the submitted values back to the template
        # and modify index.html to set the 'value' attribute of inputs and make options selected
        # Let's add submitted_form_data to the context
        return render_template('index.html',
                               dropdown_data=dropdown_data_for_template,
                               numeric_limits=numeric_limits,
                               prediction_result=formatted_price,
                               submitted_form_data=user_input_raw # Pass raw input back to repopulate form
                              )


    except Exception as e:
        # Catch any other unexpected errors during the predict route
        print(f"An unexpected error occurred in /predict route: {e}")
        # Provide a more informative error message to the user if possible
        # In debug mode, traceback will be visible, in production, log the full traceback
        import traceback
        traceback.print_exc() # Print traceback to the console/logs
        return f"An unexpected error occurred during prediction: {e}", 500


@app.route('/graphs')
def graphs():
    """Renders the graphs page."""
    # Pass the list of graph filenames to the template
    return render_template('graphs.html', graph_filenames=graph_filenames)

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

if __name__ == '__main__':
    # This is for running locally. For deployment, a production server like Gunicorn or uWSGI is recommended.
    # To make the app accessible externally (e.g., on a local network), you can use host='0.0.0.0'
    # app.run(debug=True, host='0.0.0.0')
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    