import pandas as pd
import numpy as np
import fastf1
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# 1. Setup and Caching
fastf1.Cache.enable_cache("cache_folder")

# Load session - Using 2025 as a baseline for training data
session = fastf1.get_session(2025, 'China', 'R')
session.load()

# 2. Advanced Preprocessing
# We use pick_quicklaps to remove outliers (pits, SC, errors) which inflate MAE
laps = session.laps.pick_quicklaps().copy()
laps = laps[["Driver", "LapTime", "LapNumber", "Compound", "TyreLife", "Time"]].copy()
laps.dropna(subset=["LapTime"], inplace=True)
laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

# 3. Robust Weather Integration
weather = session.weather_data.copy()
# Ensure WindGust exists; if not, fallback to WindSpeed to prevent KeyError
weather_features = ["TrackTemp", "AirTemp", "WindSpeed", "WindGust"] if "WindGust" in weather.columns else ["TrackTemp", "WindSpeed"]

# merge_asof aligns weather timestamps with lap completion times accurately
laps = pd.merge_asof(laps.sort_values('Time'), 
                     weather[['Time'] + weather_features].sort_values('Time'), 
                     on='Time', 
                     direction='backward')

# 4. Feature Engineering: The "Delta Strategy"
qualifying_data = pd.DataFrame({
    "Driver": ["RUS", "ANT", "HAD", "LEC", "PIA", "NOR", "HAM", "LAW", "LIN"],
    "QualiTime_s": [78.518, 78.811, 79.303, 79.327, 79.380, 79.475, 79.478, 79.994, 81.247]
})

full_data = laps.merge(qualifying_data, on="Driver")

# CRITICAL: Target is the difference (Delta) between race lap and quali lap
# This removes the "scale" issue and focuses only on degradation/fuel/weather effects
full_data["Delta_to_Quali"] = full_data["LapTime_s"] - full_data["QualiTime_s"]

# 5. Encoding and Feature Selection
le = LabelEncoder()
full_data["Compound_Id"] = le.fit_transform(full_data["Compound"].astype(str))

# WindSpeed and WindGust are crucial for high-speed stability in Australia
base_features = ["QualiTime_s", "LapNumber", "TyreLife", "Compound_Id"]
weather_possible = ["TrackTemp", "AirTemp", "WindSpeed", "WindGust"]

features = base_features + [c for c in weather_possible if c in full_data.columns]

X = full_data[features]
y = full_data["Delta_to_Quali"]

# 6. Training with MAE-Optimized XGBoost
# Use a lower learning rate and deeper trees to capture subtle wind/tyre interactions
split = int(len(X) * 0.8)
model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=600,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.9,
    tree_method="hist"
)

model.fit(X.iloc[:split], y.iloc[:split], eval_set=[(X.iloc[split:], y.iloc[split:])], verbose=False)

# 7. Final Race Prediction (Simulating Lap 30 conditions)
prediction_input = qualifying_data.copy()
prediction_input["LapNumber"] = 30
prediction_input["TyreLife"] = 10
prediction_input["Compound_Id"] = le.transform(["MEDIUM"])[0] if "MEDIUM" in le.classes_ else 0
for col in ["TrackTemp", "AirTemp", "WindSpeed", "WindGust"]:
    if col in full_data.columns:
        prediction_input[col] = full_data[col].mean()

# Predict the DELTA and add it back to the baseline Quali time
predicted_deltas = model.predict(prediction_input[features])
prediction_input["PredictedRaceLap"] = prediction_input["QualiTime_s"] + predicted_deltas

# 8. Output Results
print("\n🏁 2026 CHINA PREDICTION (Delta-Optimized Model) 🏁")
print(prediction_input[["Driver", "PredictedRaceLap"]].sort_values("PredictedRaceLap").to_string(index=False))

# Calculate the actual MAE based on the Delta prediction
final_mae = mean_absolute_error(y.iloc[split:], model.predict(X.iloc[split:]))
print(f"\n📉 Achieved MAE on Delta: {final_mae:.4f}s")