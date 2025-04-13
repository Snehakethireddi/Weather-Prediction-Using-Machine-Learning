from flask import Flask, render_template, request
from model import predict_weather

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    rainfall = float(request.form['rainfall'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])

    data = [rainfall, temperature, humidity, wind_speed]
    prediction = predict_weather(data)

    return render_template('index.html', prediction_text=f"Predicted Weather: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
