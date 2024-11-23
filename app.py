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
            # Retrieve form data
            form_data = {
                'age': float(request.form.get('age', 0)),
                'sex': float(request.form.get('sex', 0)),
                'chest pain type': request.form.get('chest pain type'),
                'resting blood pressure': float(request.form.get('resting blood pressure')),
                'serum cholestoral': float(request.form.get('serum cholestoral')),
                'fasting blood sugar': float(request.form.get('fasting blood sugar')),
                'resting electrocardiographic results ': request.form.get('resting electrocardiographic results '),
                'maximum heart rate achieved': float(request.form.get('maximum heart rate achieved')),
                'exercise induced angina': float(request.form.get('exercise induced angina')),
                'oldpeak': float(request.form.get('oldpeak')),
                ' slope of the peak': request.form.get(' slope of the peak'),
                'colored by flourosopy': float(request.form.get('colored by flourosopy')),
                'thal': request.form.get('thal'),
            }

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

