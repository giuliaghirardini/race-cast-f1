import os
import emoji
import fastf1

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# =======================================
# USER SETTINGS
# =======================================
gp_name = "Australia"
past_year_session = 2025
country_name = 'AUSTRALIA'
sprint_race = False

# =======================================
# PATH AND SET UP CACHE
# =======================================
# Cache
fastf1.Cache.enable_cache("cache_folder")

# Path 
path = os.path.dirname(os.path.abspath(__file__))
quali_data_path = os.path.join(os.path.join(path, "data"), 'qualifying_times.json')
quali_sprint_data_path = os.path.join(os.path.join(path, "data"), 'qualifying_sprint_times.json')

# Load session 
session = fastf1.get_session(past_year_session, gp_name, 'R')
session.load()

# =======================================
# POST PROCESSING
# =======================================

# Use pick_quicklaps to remove outliers which alter MAE
laps = session.laps.pick_quicklaps().copy()
laps = laps[["Driver", "LapTime", "LapNumber", "Compound", "TyreLife", "Time"]].copy()
laps.dropna(subset=["LapTime"], inplace=True)
laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

# Weather data
weather = session.weather_data.copy()
# Ensure WindGust exists; if not, fallback to WindSpeed to prevent KeyError
weather_features = ["TrackTemp", "AirTemp", "WindSpeed", "WindGust"] if "WindGust" in weather.columns else ["TrackTemp", "WindSpeed"]

# Align weather timestamps with lap completion times accurately
laps = pd.merge_asof(laps.sort_values('Time'), 
                     weather[['Time'] + weather_features].sort_values('Time'), 
                     on='Time', 
                     direction='backward')

# Extract from json 
quali_data = pd.read_json(quali_data_path)
quali_data.rename(columns={'BestQualiTimeSeconds': 'QualifyingTime (s)'}, inplace=True)
if sprint_race :
    quali_sprint_data = pd.read_json(quali_sprint_data_path)
    quali_sprint_data.rename(columns={'BestQualiTimeSeconds': 'QualifyingTime (s)'}, inplace=True)

qualifying_data = pd.DataFrame({
    "Driver": quali_data['Driver'].values,
    "QualiTime_s": quali_data['QualifyingTime (s)'].values,
    "SprintQualifyingTime (s)": [quali_sprint_data['QualifyingTime (s)'][i] for i in range(len(quali_sprint_data))] if sprint_race else None
})

full_data = qualifying_data.merge(laps, on="Driver")

# =======================================
# ENGINEERING FEATURES
# =======================================

### Target is the difference (Delta) between race lap and quali lap
# This removes the "scale" issue and focuses only on degradation/fuel/weather effects
full_data["Delta_to_Quali"] = full_data["LapTime_s"] - full_data["QualiTime_s"]

### Fuel Burn (Linear weight reduction)
full_data["Fuel_Burn_Effect"] = full_data["LapNumber"] * -0.033 

### Tyre Stress (Exponential degradation for high-speed tracks like Suzuka)
# This helps the model "see" the cliff where tyres suddenly lose grip
full_data["Tyre_Stress"] = full_data["TyreLife"] ** 2

### Weather Interaction (Wind hitting a hot track)
# Sometimes wind speed matters more when the track is already slippery/hot
full_data["Track_Wind_Interaction"] = full_data["TrackTemp"] * full_data["WindSpeed"]

# =======================================
# ENCODING AND FEATURE SELECTION
# =======================================
le = LabelEncoder()
full_data["Compound_Id"] = le.fit_transform(full_data["Compound"].astype(str))

# WindSpeed and WindGust are crucial for high-speed stability
base_features = ["QualiTime_s", "LapNumber", "TyreLife", "Compound_Id", 
    "Fuel_Burn_Effect", "Tyre_Stress", "Track_Wind_Interaction"]
weather_possible = ["TrackTemp", "AirTemp", "WindSpeed", "WindGust"]

features = base_features + [c for c in weather_possible if c in full_data.columns]

X = full_data[features]
y = full_data["Delta_to_Quali"]

# =======================================
# TRAINING 
# =======================================

### XGBRegressor stams for eXtreme Gradient Boosting, a powerful ensemble method that can capture complex patterns in data.
# Use a lower learning rate and deeper trees to capture subtle wind/tyre interactions
split = int(len(X) * 0.8)
model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    tree_method="hist",
    n_estimators=2000,          # Increased to allow for very small steps
    learning_rate=0.008,        # Slower learning for higher precision
    max_depth=5,                # Reduced depth to prevent overfitting on specific driver "hiccups"
    subsample=0.7,              # More aggressive sampling to handle outliers (traffic/gusts)
    colsample_bytree=0.7,       # Force trees to ignore some features to find hidden patterns

    gamma=0.2,                  # Minimum loss reduction to make a split (prevents tiny, useless branches)
    min_child_weight=5,         # Prevents the model from creating nodes for just 1 or 2 weird laps
    reg_lambda=2,               # L2 regularization (keeps weights small/stable)
    reg_alpha=0.5,              # L1 regularization (can zero out totally useless noise)
    
    early_stopping_rounds=100,  # Increased to give the slow learning rate time to converge
    random_state=39
)

model.fit(X.iloc[:split], y.iloc[:split], eval_set=[(X.iloc[split:], y.iloc[split:])], verbose=False)

# =======================================
# PREDICTION
# =======================================
prediction_input = qualifying_data.copy()

# Basic Race conditions
prediction_input["LapNumber"] = 56
prediction_input["TyreLife"] = 10
prediction_input["Fuel_Burn_Effect"] = 30 * -0.033
prediction_input["Tyre_Stress"] = 10 ** 2

# Handle Categorical Compound
try:
    prediction_input["Compound_Id"] = le.transform(["MEDIUM"])[0]
except:
    prediction_input["Compound_Id"] = 1 

# Assign the weather values from your training data averages
# You must do this BEFORE calculating the interaction
for col in ["TrackTemp", "AirTemp", "WindSpeed", "WindGust"]:
    if col in full_data.columns:
        prediction_input[col] = full_data[col].mean()
    else:
        # Fallback in case a specific weather metric is missing
        prediction_input[col] = 0.0

# Now that 'TrackTemp' and 'WindSpeed' EXIST in prediction_input, 
# you can calculate the interaction feature
prediction_input["Track_Wind_Interaction"] = (
    prediction_input["TrackTemp"] * prediction_input["WindSpeed"]
)

# Predict using the exact same features list used in training
predicted_deltas = model.predict(prediction_input[features])
prediction_input["PredictedRaceLap"] = prediction_input["QualiTime_s"] + predicted_deltas

# =======================================
# OUTPUT 
# =======================================
prediction_input = prediction_input.reset_index(drop=True)
prediction_input.index = prediction_input.index + 1

# Add medals for top 3
medals = {1: emoji.emojize(":trophy:"), 2: "🥈", 3: "🥉"}
prediction_input[" "] = prediction_input.index.map(lambda i: medals.get(i, ""))

print(f"\n🏁 Predicted 2026 {country_name} GP Winner 🏁\n")
print(prediction_input[["Driver","PredictedRaceLap", " "]].sort_values("PredictedRaceLap").to_string(index=False))

# Calculate the actual MAE based on the Delta prediction
final_mae = mean_absolute_error(y.iloc[split:], model.predict(X.iloc[split:]))
print(f"\n📉 Achieved MAE (Model error): {final_mae:.4f}s")