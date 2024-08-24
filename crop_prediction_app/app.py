from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
rf_model = joblib.load('C:/Users/HP/OneDrive/Desktop/Karatu/Predictive_Model/Crop-Recommendation/crop_prediction_app/models/random_forest_model.joblib')
gb_model = joblib.load('C:/Users/HP/OneDrive/Desktop/Karatu/Predictive_Model/Crop-Recommendation/crop_prediction_app/models/gradient_boosting_model.joblib')
lr_model = joblib.load('C:/Users/HP/OneDrive/Desktop/Karatu/Predictive_Model/Crop-Recommendation/crop_prediction_app/models/logistic_regression_model.joblib')
svm_model = joblib.load('C:/Users/HP/OneDrive/Desktop/Karatu/Predictive_Model/Crop-Recommendation/crop_prediction_app/models/svm_model.joblib')

# Load the fitted scaler
scaler = joblib.load('C:/Users/HP/OneDrive/Desktop/Karatu/Predictive_Model/Crop-Recommendation/crop_prediction_app/scaler/scaler.joblib')

# Home route to display input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission and display result
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    
    # Convert form data to numpy array
    features = np.array([[float(form_data['Nitrogen']),
                          float(form_data['Phosphorus']),
                          float(form_data['Potassium']),
                          float(form_data['Temperature']),
                          float(form_data['Humidity']),
                          float(form_data['pH_Value']),
                          float(form_data['Rainfall'])]])
    
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Get the selected model from the form
    selected_model = form_data['model']
    
    # Choose the appropriate model based on user input
    if selected_model == 'Random Forest':
        model = rf_model
    elif selected_model == 'Gradient Boosting':
        model = gb_model
    elif selected_model == 'Logistic Regression':
        model = lr_model
    elif selected_model == 'SVM':
        model = svm_model
    else:
        return '<h1>Invalid model selected</h1>'
    
    # Make the prediction
    prediction = model.predict(features_scaled)
    
    # Render the result.html template with prediction and model data
    return render_template('result.html', prediction=prediction[0], model=selected_model)

# Start the Flask server
if __name__ == "__main__":
    app.run(host='192.168.0.129', port=3455, debug=True, use_reloader=False) #Change the Host to ur local device
