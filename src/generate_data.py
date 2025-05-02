import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
regions = ["Mumbai", "Chennai", "Jaipur", "Delhi", "Pune", "Bangalore", "Hyderabad", 
           "Vijayawada", "Odisha", "Guntur", "Kolkata", "Ahmedabad", "Lucknow", 
           "Chandigarh", "Bhopal", "Patna"]
start_date = datetime(2020, 1, 1)
days = 365 * 3

# Generate dates
dates = [start_date + timedelta(days=i) for i in range(days)]

# Generate data
data = []
for region in regions:
    temp_series = []
    humidity_series = []
    pressure_series = []
    rainfall_series = []
    
    for i, date in enumerate(dates):
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        # Rainfall with high "No Rain" probability
        if region in ["Mumbai", "Kolkata"]:
            monsoon_peak = 300 * (np.sin(2 * np.pi * (day_of_year - 180) / 365) + 1) / 2 if 6 <= month <= 9 else 0
            rainfall = min(400, max(0, monsoon_peak + np.random.normal(0, 3))) if np.random.rand() > 0.35 else 0
        elif region == "Chennai":
            monsoon_peak = 250 * (np.sin(2 * np.pi * (day_of_year - 300) / 365) + 1) / 2 if 10 <= month <= 12 else 0
            rainfall = min(350, max(0, monsoon_peak + np.random.normal(0, 2))) if np.random.rand() > 0.45 else 0
        elif region in ["Vijayawada", "Guntur"]:
            monsoon_peak = 250 * (np.sin(2 * np.pi * (day_of_year - 210) / 365) + 1) / 2 if 7 <= month <= 10 else 0
            rainfall = min(350, max(0, monsoon_peak + np.random.normal(0, 2))) if np.random.rand() > 0.45 else 0
        elif region == "Odisha":
            monsoon_peak = 300 * (np.sin(2 * np.pi * (day_of_year - 180) / 365) + 1) / 2 if 6 <= month <= 9 else 0
            rainfall = min(400, max(0, monsoon_peak + np.random.normal(0, 3))) if np.random.rand() > 0.35 else 0
        else:
            monsoon_peak = 150 * (np.sin(2 * np.pi * (day_of_year - 200) / 365) + 1) / 2 if 7 <= month <= 9 else 0
            rainfall = min(200, max(0, monsoon_peak + np.random.normal(0, 2))) if np.random.rand() > 0.65 else 0

        # Temperature
        base_temp = (25 if region == "Mumbai" else 28 if region == "Chennai" else
                     30 if region in ["Jaipur", "Delhi"] else 26 if region in ["Bangalore", "Pune"] else 22)
        temp = base_temp + np.random.normal(0, 0.2)

        # Humidity
        base_humidity = 70 if region in ["Mumbai", "Kolkata", "Chennai"] else 60
        humidity = min(90, max(40, base_humidity + rainfall / 40 + np.random.normal(0, 0.5)))

        # Pressure (hPa)
        pressure = 1013 + np.random.normal(0, 1) - (rainfall / 80)

        # Store for calculations
        temp_series.append(temp)
        humidity_series.append(humidity)
        pressure_series.append(pressure)
        rainfall_series.append(rainfall)
        
        # Features
        temp_rolling7 = np.mean(temp_series[-7:]) if i >= 6 else np.mean(temp_series)
        humidity_prev = humidity_series[i-1] if i > 0 else humidity
        humidity_lag2 = humidity_series[i-2] if i > 1 else humidity
        humidity_rolling7 = np.mean(humidity_series[-7:]) if i >= 6 else np.mean(humidity_series)
        pressure_rolling3 = np.mean(pressure_series[-3:]) if i >= 2 else np.mean(pressure_series)
        temp_humidity = temp * humidity / 100
        temp_pressure = temp * pressure / 1000
        rainfall_lag3 = rainfall_series[i-3] if i > 2 else 0
        rainfall_rolling3 = np.mean(rainfall_series[-3:]) if i >= 2 else np.mean(rainfall_series)

        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Region": region,
            "Rainfall": round(rainfall, 1),
            "Temperature": round(temp, 1),
            "Humidity": round(humidity, 1),
            "Humidity_prev": round(humidity_prev, 1),
            "Humidity_Lag2": round(humidity_lag2, 1),
            "Humidity_Rolling7": round(humidity_rolling7, 1),
            "Temp_Rolling7": round(temp_rolling7, 1),
            "Temp_Humidity": round(temp_humidity, 1),
            "Pressure": round(pressure, 1),
            "Pressure_Rolling3": round(pressure_rolling3, 1),
            "Temp_Pressure": round(temp_pressure, 1),
            "Rainfall_Lag3": round(rainfall_lag3, 1),
            "Rainfall_Rolling3": round(rainfall_rolling3, 1)
        })

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv("C:/Users/Sodi Lepakshi/batch12/data/historical_rainfall.csv", index=False)
print(f"Generated {len(df)} rows of data. Saved to C:/Users/Sodi Lepakshi/batch12/data/historical_rainfall.csv")