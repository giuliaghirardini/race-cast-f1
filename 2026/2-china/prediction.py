import fastf1
import pandas as pd
import numpy as np
import emoji
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# Load FastF1 2024 Australian GP race session
session_2025 = fastf1.get_session(2025, 'China', 'R')
session_2025.load()

# Extract Lap times
laps_2025 = session_2025.laps[["Driver", "LapTime"]].copy()
laps_2025.dropna(subset=["LapTime"], inplace=True)
laps_2025["LapTime (s)"] = laps_2025["LapTime"].dt.total_seconds()

# Qualifying data
qualifying_2026 = pd.DataFrame({
    "Driver": ["ANT", "RUS", "HAM", "LEC", "PIA", "NOR", "GAS", "VER", "HAD", "BEA"],
    "QualifyingTime (s)": [92.064, 92.286, 92.415, 92.428, 92.550, 92.608, 92.873, 93.002, 93.121, 93.292]
})

merged_data = qualifying_2026.merge(laps_2025)

X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError('Dataset is empty after preprocessing. Check data sources!')

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2026[["QualifyingTime (s)"]])
qualifying_2026["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2026 = qualifying_2026.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)

drivers = {
    "NOR": {"DriverNumber": 1,  "DriverName": "Lando Norris"},
    "VER": {"DriverNumber": 3,  "DriverName": "Max Verstappen"},
    "HAD": {"DriverNumber": 6,  "DriverName": "Isack Hadjar"},
    "ANT": {"DriverNumber": 12, "DriverName": "Andrea Kimi Antonelli"},
    "GAS": {"DriverNumber": 10, "DriverName": "Pierre Gasly"},
    "COL": {"DriverNumber": 43, "DriverName": "Franco Colapinto"},
    "OCO": {"DriverNumber": 31, "DriverName": "Esteban Ocon"},
    "BEA": {"DriverNumber": 87, "DriverName": "Oliver Bearman"},
    "ALO": {"DriverNumber": 14, "DriverName": "Fernando Alonso"},
    "STR": {"DriverNumber": 18, "DriverName": "Lance Stroll"},
    "SAI": {"DriverNumber": 55, "DriverName": "Carlos Sainz"},
    "ALB": {"DriverNumber": 23, "DriverName": "Alexander Albon"},
    "HUL": {"DriverNumber": 27, "DriverName": "Nico Hülkenberg"},
    "HAM": {"DriverNumber": 44, "DriverName": "Lewis Hamilton"},
    "LEC": {"DriverNumber": 16, "DriverName": "Charles Leclerc"},
    "PIA": {"DriverNumber": 81, "DriverName": "Oscar Piastri"},
    "LAW": {"DriverNumber": 30, "DriverName": "Liam Lawson"},
    "RUS": {"DriverNumber": 63, "DriverName": "George Russell"},
    "PER": {"DriverNumber": 11, "DriverName": "Sergio Pérez"},
    "BOT": {"DriverNumber": 77, "DriverName": "Valtteri Bottas"},
    "BOR": {"DriverNumber": 5,  "DriverName": "Gabriel Bortoleto"},
    "LIN": {"DriverNumber": 41, "DriverName": "Arvid Lindblad"}
}

# Print final predictions
print("\n🏁 Predicted 2026 China GP Winner 🏁\n")

# Add driver names
qualifying_2026["DriverName"] = qualifying_2026["Driver"].map(lambda code: drivers.get(code, {}).get("DriverName", code))

# Add position (1-based index)
qualifying_2026 = qualifying_2026.reset_index(drop=True)
qualifying_2026.index = qualifying_2026.index + 1

# Add medals for top 3
medals = {1: emoji.emojize(":trophy:"), 2: "🥈", 3: "🥉"}
qualifying_2026["Medal"] = qualifying_2026.index.map(lambda i: medals.get(i, ""))

# Print with medals
print(qualifying_2026[["Driver", "DriverName", "PredictedRaceTime (s)", "Medal"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n📉 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
