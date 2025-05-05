import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained model
model_path = "models/rainfall_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# OpenWeather API key (set via environment variable)
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Placeholder: Load test data (replace with your actual test data)
# Example: X_test, y_test = load_test_data("path/to/test_data.csv")
# For now, simulate test data
np.random.seed(42)
regions = ["Mumbai", "Chennai", "Jaipur", "Delhi", "Pune", "Bangalore", "Hyderabad", 
           "Vijayawada", "Odisha", "Guntur", "Kolkata", "Ahmedabad", "Lucknow", 
           "Chandigarh", "Bhopal", "Patna"]
features = [
    "Temperature", "Humidity", "Humidity_prev", "Humidity_Lag2", "Humidity_Rolling7",
    "Temp_Rolling7", "Temp_Humidity", "Pressure", "Pressure_Rolling3", "Temp_Pressure",
    "Rainfall_prev", "Rainfall_prev_2", "Rainfall_Lag3", "Rainfall_Rolling7",
    "Month", "DayOfYear", "Sin_Month", "Cos_Month", "Sin_Day", "Cos_Day"
] + [f"Region_{r}" for r in regions]
X_test = pd.DataFrame(np.random.rand(1000, len(features)), columns=features)
y_test = np.random.rand(1000) * 400  # Simulated rainfall in mm
# Simulate city assignments for test data
city_assignments = np.random.choice(regions, size=1000)

# Helper function to get current weather data from OpenWeather API
def get_weather_data(region):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={region}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"]
    }

# Helper function to categorize rainfall
def categorize_rainfall(rainfall):
    if rainfall == 0:
        return "No Rain"
    elif rainfall < 2.5:
        return "Light Rain"
    elif rainfall < 7.5:
        return "Moderate Rain"
    else:
        return "Heavy Rain"

# Generate Feature Importance Plot
def generate_feature_importance_plot():
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(features, importances)
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Importance in XGBoost Model")
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode("utf-8")

# Generate Predicted vs. Actual Rainfall Plot
def generate_prediction_plot():
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 400], [0, 400], 'r--')
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Predicted Rainfall (mm)")
    plt.title("Predicted vs. Actual Rainfall")
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode("utf-8")

# Generate Categorical Accuracy by City Plot
def generate_category_accuracy_plot():
    accuracies = {}
    for city in regions:
        # Filter test data for the city
        city_mask = city_assignments == city
        if city_mask.sum() == 0:
            accuracies[city] = 0
            continue
        X_city = X_test[city_mask]
        y_city = y_test[city_mask]
        y_pred = model.predict(X_city)
        
        # Categorize predictions and actuals
        categories_true = np.array([categorize_rainfall(y) for y in y_city])
        categories_pred = np.array([categorize_rainfall(y) for y in y_pred])
        
        # Compute accuracy
        accuracy = np.mean(categories_true == categories_pred)
        accuracies[city] = accuracy * 100  # Convert to percentage
    
    plt.figure(figsize=(12, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("City")
    plt.ylabel("Categorical Accuracy (%)")
    plt.title("Categorical Accuracy by City")
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode("utf-8")

@app.route("/")
def index():
    # Generate charts
    feature_importance_plot = generate_feature_importance_plot()
    prediction_plot = generate_prediction_plot()
    category_accuracy_plot = generate_category_accuracy_plot()
    
    return render_template(
        "index.html",
        feature_importance_plot=feature_importance_plot,
        prediction_plot=prediction_plot,
        category_accuracy_plot=category_accuracy_plot
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        region = data.get("region")

        # Get current weather data
        weather = get_weather_data(region)
        if not weather:
            return jsonify({"error": "Unable to fetch weather data"}), 500

        # Prepare features for prediction
        current_date = datetime.now()
        month = current_date.month
        day_of_year = current_date.timetuple().tm_yday

        features_dict = {
            "Temperature": weather["temperature"],
            "Humidity": weather["humidity"],
            "Humidity_prev": weather["humidity"],
            "Humidity_Lag2": weather["humidity"],
            "Humidity_Rolling7": weather["humidity"],
            "Temp_Rolling7": weather["temperature"],
            "Temp_Humidity": weather["temperature"] * weather["humidity"],
            "Pressure": weather["pressure"],
            "Pressure_Rolling3": weather["pressure"],
            "Temp_Pressure": weather["temperature"] * weather["pressure"],
            "Rainfall_prev": 0,
            "Rainfall_prev_2": 0,
            "Rainfall_Lag3": 0,
            "Rainfall_Rolling7": 0,
            "Month": month,
            "DayOfYear": day_of_year,
            "Sin_Month": np.sin(2 * np.pi * month / 12),
            "Cos_Month": np.cos(2 * np.pi * month / 12),
            "Sin_Day": np.sin(2 * np.pi * day_of_year / 365),
            "Cos_Day": np.cos(2 * np.pi * day_of_year / 365)
        }

        # Add region as one-hot encoded features
        for r in regions:
            features_dict[f"Region_{r}"] = 1 if r == region else 0

        # Convert to DataFrame
        feature_df = pd.DataFrame([features_dict])

        # Predict rainfall
        prediction = model.predict(feature_df)[0]
        prediction = max(0, prediction)
        category = categorize_rainfall(prediction)

        return jsonify({
            "region": region,
            "rainfall_prev": 0,
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "prediction": round(prediction, 1),
            "category": category
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)