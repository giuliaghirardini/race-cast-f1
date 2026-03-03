import fastf1
import os
import pandas as pd
import numpy as np
import emoji
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
cache_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), '__cache_folder')
os.makedirs(cache_folder, exist_ok=True)
fastf1.Cache.enable_cache(cache_folder)

# Load FastF1 2024 Australian GP race session
session_2024 = fastf1.get_session(2024, 'Japan', 'R')
session_2024.load()

# Extract Lap times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]

# Qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "ANT", "HAD", "HAM", "ALB", "BEA"],
    "QualifyingTime (s)": [86.983, 86.995, 87.027, 87.299, 87.318, 87.555, 87.569, 87.610, 87.615, 87.867]
})

merged_data = qualifying_2025.merge(sector_times_2024)

# Driver wet performance
driver_wet_performance = {
    "ALB": 0.978120,
    "ALO": 0.972655,
    "BOT": 0.982052,
    "GAS": 0.978832,
    "HAM": 0.976464,
    "LEC": 0.975862,
    "MAG": 0.989983,
    "NOR": 0.978179,
    "OCO": 0.981810,
    "PER": 0.998904,
    "RUS": 0.968678,
    "SAI": 0.978754,
    "STR": 0.979857,
    "TSU": 0.996338,
    "VER": 0.975196,
    "ZHO": 0.987774
}

merged_data["WetPerformanceFactor"] = merged_data["Driver"].map(driver_wet_performance)

# Forecasted weather data for Suzuka using OpenWeatherMap API
API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your actual API key
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?q=Suzuka,jp&appid={API_KEY}&units=metric"

# Fetch weather data
response = requests.get(weather_url)
weather_data = response.json()

# Extract the relevant weather data for the race (Sunday at 2pm local time)
forecast_time = "2025-03-30 14:00:00"
forecast_data = None
for forecast in weather_data["list"]:
    if forecast["dt_txt"] == forecast_time:
        forecast_data = forecast
        break

# Extract the weather features (rain probability, temperature)
if forecast_data:
    rain_probability = forecast_data["pop"]       # Rain probability (0 to 1)
    temperature = forecast_data["main"]["temp"]   # Temperature in Celsius
else:
    rain_probability = 0.0  # Default if no data is found
    temperature = 20

merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

X = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "WetPerformanceFactor"]]
y = merged_data.merge(laps_2024.groupby("Driver")["LapTime (s)"].mean())

if X.shape[0] == 0:
    raise ValueError('Dataset is empty after preprocessing. Check data sources!')

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_race_times = model.predict(X)
qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)

drivers = {
    "ALB": {"DriverNumber": 23, "DriverName": "Alexander Albon"},
    "ALO": {"DriverNumber": 14, "DriverName": "Fernando Alonso"},
    "ANT": {"DriverNumber": 7,  "DriverName": "Kimi Antonelli"},
    "BEA": {"DriverNumber": 87, "DriverName": "Oliver Bearman"},
    "BOR": {"DriverNumber": 10, "DriverName": "Gabriel Bortoleto"},
    "BOT": {"DriverNumber": 77, "DriverName": "Valtteri Bottas"},
    "COL": {"DriverNumber": 43, "DriverName": "Franco Colapinto"},
    "DEV": {"DriverNumber": 45, "DriverName": "Nyck de Vries"},
    "DOO": {"DriverNumber": 61, "DriverName": "Jack Doohan"},
    "GAS": {"DriverNumber": 10, "DriverName": "Pierre Gasly"},
    "HAD": {"DriverNumber": 20, "DriverName": "Isack Hadjar"},
    "HAM": {"DriverNumber": 44, "DriverName": "Lewis Hamilton"},
    "HUL": {"DriverNumber": 27, "DriverName": "Nico Hülkenberg"},
    "LAW": {"DriverNumber": 30, "DriverName": "Liam Lawson"},
    "LEC": {"DriverNumber": 16, "DriverName": "Charles Leclerc"},
    "MAG": {"DriverNumber": 20, "DriverName": "Kevin Magnussen"},
    "NOR": {"DriverNumber": 4,  "DriverName": "Lando Norris"},
    "OCO": {"DriverNumber": 31, "DriverName": "Esteban Ocon"},
    "PER": {"DriverNumber": 11, "DriverName": "Sergio Pérez"},
    "PIA": {"DriverNumber": 81, "DriverName": "Oscar Piastri"},
    "RIC": {"DriverNumber": 3,  "DriverName": "Daniel Ricciardo"},
    "RUS": {"DriverNumber": 63, "DriverName": "George Russell"},
    "SAI": {"DriverNumber": 55, "DriverName": "Carlos Sainz"},
    "SAR": {"DriverNumber": 2,  "DriverName": "Logan Sargeant"},
    "STR": {"DriverNumber": 18, "DriverName": "Lance Stroll"},
    "TSU": {"DriverNumber": 22, "DriverName": "Yuki Tsunoda"},
    "VER": {"DriverNumber": 1,  "DriverName": "Max Verstappen"},
    "ZHO": {"DriverNumber": 24, "DriverName": "Zhou Guanyu"}
}

# Print final predictions
print("\n🏁 Predicted 2025 Chinese GP Winner 🏁\n")

# Add driver names
qualifying_2025["DriverName"] = qualifying_2025["Driver"].map(lambda code: drivers.get(code, {}).get("DriverName", code))

# Add position (1-based index)
qualifying_2025 = qualifying_2025.reset_index(drop=True)
qualifying_2025.index = qualifying_2025.index + 1

# Add medals for top 3
medals = {1: emoji.emojize(":trophy:"), 2: "🥈", 3: "🥉"}
qualifying_2025["Medal"] = qualifying_2025.index.map(lambda i: medals.get(i, ""))

# Print with medals
print(qualifying_2025[["Driver", "DriverName", "PredictedRaceTime (s)", "Medal"]])


# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n📉 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
