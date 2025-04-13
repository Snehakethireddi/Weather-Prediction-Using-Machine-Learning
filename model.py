import numpy as np
import joblib

# Load model and other components
model = joblib.load('rainfall_model.pkl')
scaler = joblib.load('scaler.pkl')
labelencoder = joblib.load('labelencoder.pkl')


def predict_weather(data):
    # Convert input to array and reshape
    final_input = np.array([data])

    # Scale input
    final_input_scaled = scaler.transform(final_input)

    # Predict
    prediction = model.predict(final_input_scaled)

    # Decode label
    result = labelencoder.inverse_transform(prediction)
    return result[0]
