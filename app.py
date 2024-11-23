from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Set paths for the preprocessor and model
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
MODEL_PATH = os.path.join("artifacts", "xgboost.pkl")

# Ensure the required .pkl files exist
if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(f"Preprocessor file not found at: {PREPROCESSOR_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load the preprocessor and model at the start
try:
    with open(PREPROCESSOR_PATH, 'rb') as file:
        preprocessor = pickle.load(file)
    with open(MODEL_PATH, 'rb') as file:
        xgb_model = pickle.load(file)
    print("Preprocessor and model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading preprocessor or model: {e}")

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Retrieve form data with validation for missing values
            form_data = {
                'age': float(request.form.get('age', 0)),
                'sex': float(request.form.get('sex', 0)),  # Default to 0 if missing
                'chest pain type': request.form.get('chest pain type', 'typical angina'),  # Default to 'typical angina'
                'resting blood pressure': float(request.form.get('resting blood pressure', 0)),
                'serum cholestoral': float(request.form.get('serum cholestoral', 0)),
                'fasting blood sugar': float(request.form.get('fasting blood sugar', 0)),
                'resting electrocardiographic results ': request.form.get('resting electrocardiographic results ', 'normal'),  # Default to 'normal'
                'maximum heart rate achieved': float(request.form.get('maximum heart rate achieved', 0)),
                'exercise induced angina': float(request.form.get('exercise induced angina', 0)),
                'oldpeak': float(request.form.get('oldpeak', 0)),
                ' slope of the peak': request.form.get(' slope of the peak', 'flat'),  # Default to 'flat'
                'colored by flourosopy': float(request.form.get('colored by flourosopy', 0)),
                'thal': request.form.get('thal', 'normal')  # Default to 'normal'
            }

            # Check for missing or invalid values in form data
            for key, value in form_data.items():
                if value in [None, '', 'None']:
                    raise ValueError(f"Missing or invalid value for {key}")

            # Convert form data to DataFrame
            input_data_df = pd.DataFrame([form_data])
            print("Input DataFrame:\n", input_data_df)

            # Preprocess input data
            input_data_processed = preprocessor.transform(input_data_df)

            # Make prediction
            model_prediction = xgb_model.predict(input_data_processed)[0]
            prediction = "Yes" if model_prediction == 1 else "No"

            # Redirect to display page
            return redirect(url_for('display', res=prediction))

        except Exception as e:
            print("Error during prediction:", e)
            prediction = f"Error: Unable to process the request. Details: {e}"

    return render_template('index.html', prediction=prediction)

@app.route('/display/<res>', methods=['GET'])
def display(res):
    return render_template('display.html', res=res)

if __name__ == '__main__':
    app.run(debug=True)
