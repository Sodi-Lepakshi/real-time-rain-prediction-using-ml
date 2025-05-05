from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import requests
from datetime import datetime
import numpy as np
import logging
import os

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Log model loading
    logger.info("Loading model")

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "rainfall_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # OpenWeather API setup
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not API_KEY:
        raise ValueError("OpenWeather API key not found. Set the OPENWEATHER_API_KEY environment variable.")
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def fetch_weather(city):
        logger.info(f"Fetching weather for {city}")
        try:
            url = f"{BASE_URL}?q={city},IN&appid={API_KEY}&units=metric"
            response = requests.get(url).json()
            if response.get("cod") != 200:
                raise Exception(f"API Error: {response.get('message')}")
            rainfall = response.get("rain", {}).get("1h", 0)
            temp = response["main"]["temp"]
            humidity = response["main"]["humidity"]
            pressure = response["main"]["pressure"]
            logger.info(f"Weather for {city}: rainfall={rainfall}, temp={temp}, humidity={humidity}, pressure={pressure}")
            return rainfall, temp, humidity, pressure
        except Exception as e:
            logger.error(f"Error fetching weather for {city}: {e}")
            return 0, 25, 60, 1013

    def get_rainfall_category(rainfall):
        if rainfall < 2.5:
            return "No Rain"
        elif rainfall < 10:
            return "Light Rain"
        elif rainfall < 50:
            return "Moderate Rain"
        else:
            return "Heavy Rain"

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            region = request.json["region"]
            logger.info(f"Predicting for region: {region}")
            rainfall_prev, temp, humidity, pressure = fetch_weather(region)

            current_date = datetime.now()
            month = current_date.month
            day_of_year = current_date.timetuple().tm_yday
            sin_month = np.sin(2 * np.pi * month / 12)
            cos_month = np.cos(2 * np.pi * month / 12)
            sin_day = np.sin(2 * np.pi * day_of_year / 365)
            cos_day = np.cos(2 * np.pi * day_of_year / 365)

            input_data = pd.DataFrame({
                "Temperature": [temp],
                "Humidity": [humidity],
                "Humidity_prev": [humidity],
                "Humidity_Lag2": [humidity],
                "Humidity_Rolling7": [humidity],
                "Temp_Rolling7": [temp],
                "Temp_Humidity": [temp * humidity / 100],
                "Pressure": [pressure],
                "Pressure_Rolling3": [pressure],
                "Temp_Pressure": [temp * pressure / 1000],
                "Rainfall_prev": [rainfall_prev],
                "Rainfall_prev_2": [0],
                "Rainfall_Lag3": [0],
                "Rainfall_Rolling7": [rainfall_prev],
                "Rainfall_Rolling3": [rainfall_prev],
                "Month": [month],
                "DayOfYear": [day_of_year],
                "Sin_Month": [sin_month],
                "Cos_Month": [cos_month],
                "Sin_Day": [sin_day],
                "Cos_Day": [cos_day],
                "Region": [region]
            })
            input_data = pd.get_dummies(input_data)
            input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

            prediction = model.predict(input_data)[0]
            prediction = max(0, prediction * 0.75 if prediction < 20 else prediction)
            category = get_rainfall_category(prediction)

            response = {
                "region": region,
                "rainfall_prev": round(float(rainfall_prev), 2),
                "temperature": round(float(temp), 2),
                "humidity": round(float(humidity), 2),
                "prediction": round(float(prediction), 2),
                "category": category
            }
            logger.info(f"Raw prediction: {prediction}, Category: {category}")
            logger.info(f"Response for {region}: {response}")
            return jsonify(response)
        except Exception as e:
            error_response = {"error": str(e), "category": "Error"}
            logger.error(f"Prediction error: {e}")
            return jsonify(error_response), 500

except Exception as e:
    logger.error(f"Error initializing app: {str(e)}")
    raise

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
