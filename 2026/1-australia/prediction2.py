import pandas as pd
import numpy as np
import fastf1
import emoji
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# Load FastF1 2025 Australian GP race session
session = fastf1.get_session(2025, 'Australia', 'R')
session.load()

# Extract Lap times
laps = session.laps.pick_quicklaps() # Esclude safety car e inlap/outlap estremi
laps = laps[["Driver", "LapTime", "LapNumber", "Stint", "Compound", "TyreLife"]].copy()
laps.dropna(inplace=True)
laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

# Meteo data
weather = session.laps.get_weather_data().reset_index(drop=True)
laps = pd.concat([laps.reset_index(drop=True), weather[["TrackTemp", "AirTemp"]]], axis=1)
# Categorical variables
le = LabelEncoder()
laps["Compound_Id"] = le.fit_transform(laps["Compound"].astype(str))

# Qualifying data
qualifying_data = pd.DataFrame({
    "Driver": ["RUS", "ANT", "HAD", "LEC", "PIA", "NOR", "HAM", "LAW", "LIN"],
    "QualiTime_s": [78.518, 78.811, 79.303, 79.327, 79.380, 79.475, 79.478, 79.994, 81.247]
})

full_data = laps.merge(qualifying_data, on="Driver")

features = ["QualiTime_s", "LapNumber", "TyreLife", "Compound_Id", "TrackTemp"]
X = full_data[features]
y = full_data["LapTime_s"]

# Training with Random Forest 
# Divide the data: use the first 2/3 of the race to train the model and the last 1/3 for testing
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=150, max_depth=7, random_state=39)
model.fit(X_train, y_train)

# Prediction for the race
# --------------------------------- lap 30, TyreLife 10, Medium compound, Mean temperature ----------------------------
avg_track_temp = laps["TrackTemp"].mean()
compound_med = le.transform(["MEDIUM"])[0] if "MEDIUM" in le.classes_ else 0

prediction_input = qualifying_data.copy()
prediction_input["LapNumber"] = 30 
prediction_input["TyreLife"] = 10
prediction_input["Compound_Id"] = compound_med
prediction_input["TrackTemp"] = avg_track_temp

qualifying_data["PredictedRaceLap_s"] = model.predict(prediction_input[features])

# Rank drivers by predicted race time
qualifying_data = qualifying_data.sort_values(by="PredictedRaceLap_s").reset_index(drop=True)
qualifying_data.index += 1

print("\n🏁 Predicted 2025 Australian GP (Mean Pace) 🏁\n")
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
for idx, row in qualifying_data.iterrows():
    m = medals.get(idx, "  ")
    print(f"{idx}. {m} {row['Driver']} | Quali: {row['QualiTime_s']:.3f}s | Pred. race: {row['PredictedRaceLap_s']:.3f}s")

# Valutazione
y_pred = model.predict(X_test)
print(f"\n📉 Model Average Error (MAE): {mean_absolute_error(y_test, y_pred):.3f} secondi")

# ----------------- lap 50, TyreLife 30, Hard compound, Mean temperature ------------------------------------
avg_track_temp = laps["TrackTemp"].mean()
compound_hard = le.transform(["HARD"])[0] if "HARD" in le.classes_ else 0

prediction_input = qualifying_data.copy()
prediction_input["LapNumber"] = 58 
prediction_input["TyreLife"] = 30
prediction_input["Compound_Id"] = compound_hard
prediction_input["TrackTemp"] = avg_track_temp

qualifying_data["PredictedRaceLap_s"] = model.predict(prediction_input[features])

# Rank drivers by predicted race time
qualifying_data = qualifying_data.sort_values(by="PredictedRaceLap_s").reset_index(drop=True)
qualifying_data.index += 1

print("\n🏁 Predicted 2025 Australian GP (Mean Pace) 🏁\n")
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
for idx, row in qualifying_data.iterrows():
    m = medals.get(idx, "  ")
    print(f"{idx}. {m} {row['Driver']} | Quali: {row['QualiTime_s']:.3f}s | Pred. race: {row['PredictedRaceLap_s']:.3f}s")

# Valutazione
y_pred = model.predict(X_test)
print(f"\n📉 Model Average Error (MAE): {mean_absolute_error(y_test, y_pred):.3f} secondi")