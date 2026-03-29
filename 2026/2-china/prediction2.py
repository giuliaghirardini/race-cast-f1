import pandas as pd
import numpy as np
import fastf1
import emoji
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# 1. Setup and Caching
fastf1.Cache.enable_cache("cache_folder")

# Load session data
session = fastf1.get_session(2025, 'China', 'R')
session.load()

# 2. Extract and Clean Lap Data
laps = session.laps.pick_quicklaps().copy()
laps = laps[["Driver", "LapTime", "LapNumber", "Stint", "Compound", "TyreLife"]].copy()
laps.dropna(inplace=True)
laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

# 3. Handle Weather Data Safely
# weather_data is indexed by time, we reset to align with laps
weather = session.laps.get_weather_data().reset_index(drop=True)

# List of columns we WANT. We check if they actually exist to avoid KeyError.
requested_weather = ["TrackTemp", "AirTemp", "WindSpeed", "WindGust"]
available_weather = [col for col in requested_weather if col in weather.columns]

# If WindGust is missing, we create it as a copy of WindSpeed so the model doesn't crash
if "WindGust" not in weather.columns and "WindSpeed" in weather.columns:
    weather["WindGust"] = weather["WindSpeed"]
    available_weather.append("WindGust")

# Concatenate weather with lap data
laps = pd.concat([laps.reset_index(drop=True), weather[available_weather]], axis=1)

# 4. Encode Categorical Variables
le = LabelEncoder()
laps["Compound_Id"] = le.fit_transform(laps["Compound"].astype(str))

# 5. Qualifying Baseline Data (2026 Prediction context)
qualifying_data = pd.DataFrame({
    "Driver": ["RUS", "ANT", "HAD", "LEC", "PIA", "NOR", "HAM", "LAW", "LIN"],
    "QualiTime_s": [78.518, 78.811, 79.303, 79.327, 79.380, 79.475, 79.478, 79.994, 81.247]
})

full_data = laps.merge(qualifying_data, on="Driver")

# 6. Model Training
# Included WindSpeed and WindGust in features to match your prediction logic
features = ["QualiTime_s", "LapNumber", "TyreLife", "Compound_Id", "TrackTemp", "WindSpeed", "WindGust"]
X = full_data[features]
y = full_data["LapTime_s"]

# Split data chronologically (80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=150, max_depth=7, random_state=39)
model.fit(X_train, y_train)

# 7. Prediction for Lap 30 (Medium Compound)
avg_track_temp = laps["TrackTemp"].mean()
avg_wind_speed = laps["WindSpeed"].mean()
avg_wind_gust = laps["WindGust"].mean()
compound_med = le.transform(["MEDIUM"])[0] if "MEDIUM" in le.classes_ else 0

prediction_input = qualifying_data.copy()
prediction_input["LapNumber"] = 30 
prediction_input["TyreLife"] = 10
prediction_input["Compound_Id"] = compound_med
prediction_input["TrackTemp"] = avg_track_temp
prediction_input["WindSpeed"] = avg_wind_speed
prediction_input["WindGust"] = avg_wind_gust

qualifying_data["PredictedRaceLap_s"] = model.predict(prediction_input[features])

# Output Results for Lap 30
qualifying_data = qualifying_data.sort_values(by="PredictedRaceLap_s").reset_index(drop=True)
qualifying_data.index += 1

print("\n🏁 Predicted 2025 Chinese GP (Lap 30 - Medium Pace) 🏁\n")
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
for idx, row in qualifying_data.iterrows():
    m = medals.get(idx, "  ")
    print(f"{idx}. {m} {row['Driver']} | Quali: {row['QualiTime_s']:.3f}s | Pred. race: {row['PredictedRaceLap_s']:.3f}s")

# 8. Evaluation
y_pred = model.predict(X_test)
print(f"\n📉 Model Average Error (MAE): {mean_absolute_error(y_test, y_pred):.3f} seconds")

# 9. Prediction for Lap 58 (Hard Compound)
# Note: Resetting qualifying_data to original drivers for the second run
qualifying_data_final = qualifying_data.drop(columns=["PredictedRaceLap_s"])
compound_hard = le.transform(["HARD"])[0] if "HARD" in le.classes_ else 0

prediction_input_final = qualifying_data_final.copy()
prediction_input_final["LapNumber"] = 58 
prediction_input_final["TyreLife"] = 30
prediction_input_final["Compound_Id"] = compound_hard
prediction_input_final["TrackTemp"] = avg_track_temp
prediction_input_final["WindSpeed"] = avg_wind_speed
prediction_input_final["WindGust"] = avg_wind_gust

qualifying_data_final["PredictedRaceLap_s"] = model.predict(prediction_input_final[features])

# Output Results for Lap 58
qualifying_data_final = qualifying_data_final.sort_values(by="PredictedRaceLap_s").reset_index(drop=True)
qualifying_data_final.index = range(1, len(qualifying_data_final) + 1)

print("\n🏁 Predicted 2025 Chinese GP (Lap 58 - Hard/End of Race Pace) 🏁\n")
for idx, row in qualifying_data_final.iterrows():
    m = medals.get(idx, "  ")
    print(f"{idx}. {m} {row['Driver']} | Quali: {row['QualiTime_s']:.3f}s | Pred. race: {row['PredictedRaceLap_s']:.3f}s")