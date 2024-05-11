from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('stroke_prediction_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting input features from the form
        features = [float(x) for x in request.form.values()]
        # Converting the features to a numpy array
        input_features = np.array(features).reshape(1, -1)
        # Making prediction
        prediction = model.predict(input_features)
        if prediction == 1:
            result = 'The patient is likely to have a stroke.'
        else:
            result = 'The patient is not likely to have a stroke.'
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
