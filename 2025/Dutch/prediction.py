import fastf1
import os
import pandas as pd
import numpy as np
import emoji
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
cache_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), '__cache_folder')
os.makedirs(cache_folder, exist_ok=True)
fastf1.Cache.enable_cache(cache_folder)

# Load FastF1 2024 Australian GP race session
session_2024 = fastf1.get_session(2024, 'Dutch', 'R')
session_2024.load()

# Extract Lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# Qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": ["PIA", "NOR", "VER", "HAD", "RUS", "LEC", "HAM", "LAW", "SAI", "ALO"],
    "QualifyingTime (s)": [68.662, 68.674, 68.925, 69.208, 69.255, 69.340, 69.390, 69.500, 69.505, 69.630]
})

merged_data = qualifying_2025.merge(laps_2024)

X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError('Dataset is empty after preprocessing. Check data sources!')

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)

drivers = {
    "ALB": {"DriverNumber": 23, "DriverName": "Alexander Albon"},
    "ALO": {"DriverNumber": 14, "DriverName": "Fernando Alonso"},
    "ANT": {"DriverNumber": 7,  "DriverName": "Andrea Kimi Antonelli"},
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
print("\n🏁 Predicted 2025 Dutch GP Winner 🏁\n")

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
