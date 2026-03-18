import os
import fastf1
import pandas as pd
import json as js

################################################################
# FUNCTIONS
################################################################

# Function to sum lap times for a given qualifying session
def best_lap_times(laps_session, driver_numbers):
    '''Calculate best lap times for each driver in a qualifying session and map driver numbers.'''

    df_session = laps_session[['Driver', 'LapTime']].dropna(subset=['LapTime'])
    best_times = df_session.groupby('Driver')['LapTime'].min().reset_index()

    # Sort by best lap time before mapping numbers
    best_times = best_times.sort_values('LapTime').reset_index(drop=True)
    best_times['DriverNumber'] = best_times['Driver'].map(driver_numbers)

    return best_times[['Driver', 'DriverNumber', 'LapTime']].rename(columns={'LapTime': 'BestQualiTime'})

################################################################
# PATH AND DATA LOADING
################################################################

race_name = "China"
year = 2026

################################################################
# PATH AND DATA LOADING
################################################################

# Set up cache folder
cache_folder = 'cache_folder'
os.makedirs(cache_folder, exist_ok=True)
fastf1.Cache.enable_cache(cache_folder)
print("\nCache enabled successfully!")

# Set up for json output
path = os.path.dirname(os.path.abspath(__file__))
global_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(path, "data"), exist_ok=True)
data_path = os.path.join(path, "data")
print(f"\n\nGlobal path is: {global_path}")
print(f"\nBase path is: {path}")
print(f"\nData path set to: {data_path}")

# JSON file path
json_namefile = 'qualifying_sprint_times.json'
json_path = os.path.join(data_path, json_namefile)
print(f"\nJSON output will be saved to: {json_path}")

# Load session data
print("\n-------------------------------------------------------------------\n")
session = fastf1.get_session(year, race_name, 'SQ')
session.load(telemetry=False)
print("\n-------------------------------------------------------------------\n")

print("\nData loaded successfully!\n")

################################################################
# DATA EXTRACTION AND PROCESSING
################################################################

# Extract laps and split into qualifying sessions
laps = session.laps
df = laps[['Driver', 'LapTime', 'Compound']]
qualifying_sessions = dict(zip(['Q1', 'Q2', 'Q3'], laps.split_qualifying_sessions()))

driver_numbers = pd.read_json(os.path.join(global_path, "driver_names_2026.json"), orient='index').iloc[:, 0]

# Evaluate best times for each quali session (Q1, Q2, Q3)
all_results = {}

for q_name, q_data in qualifying_sessions.items():
    total_times = best_lap_times(q_data, driver_numbers).sort_values('BestQualiTime')
    total_times['BestQualiTimeSeconds'] = total_times['BestQualiTime'].dt.total_seconds()
    all_results[q_name] = total_times
    
################################################################
# PRINT
################################################################

for q_name, total_times in all_results.items():
    print(f'\n🏁 Total Sprint Qualifying Times {q_name} 🏁\n')
    total_times.index = total_times.index + 1
    print(total_times)

################################################################
# SAVE TO JSON
################################################################

# Use Q3 data for JSON output (or adjust based on your needs)
total_times = all_results.get('Q3', all_results[list(all_results.keys())[-1]])

# Save qualifying times to JSON
json_output = total_times[['Driver', 'DriverNumber', 'BestQualiTimeSeconds']].to_dict(orient='records')

with open(json_path, 'w') as f:
    js.dump(json_output, f, indent=4)
print("\n\nQualifying times saved to ../data/qualifying_sprint_times.json\n\n")