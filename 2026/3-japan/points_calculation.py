import os
import fastf1
import json as js
import numpy as np
import pandas as pd

################################################################
# PATH AND DATA LOADING
################################################################

race_name = "Japan"
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
json_namefile = 'gp_points.json'
json_path = os.path.join(data_path, json_namefile)
print(f"\nJSON output will be saved to: {json_path}")

# Load session data
print("\n-------------------------------------------------------------------\n")
race_session = fastf1.get_session(year, race_name, 'R')
sprint_session = fastf1.get_session(year, race_name, 'S')
race_session.load(telemetry=False)
sprint_session.load(telemetry=False)
print("\n-------------------------------------------------------------------\n")

print("\nData loaded successfully!\n")

################################################################
# DATA EXTRACTION AND PROCESSING
################################################################

def extract_points(session1, session2):
    # Extract points for each driver
    points_data1 = session1.results[['Abbreviation', 'Points']].dropna(subset=['Points'])
    points_data1 = points_data1.rename(columns={'Abbreviation': 'Driver'})
    points_data1['Points'] = points_data1['Points'].astype(int)
    print("Points from race session:")
    print(points_data1)

    # Extract points for each driver in the second session
    points_data2 = session2.results[['Abbreviation', 'Points']].dropna(subset=['Points'])
    points_data2 = points_data2.rename(columns={'Abbreviation': 'Driver'})
    points_data2['Points'] = points_data2['Points'].astype(int)
    print("Points from sprint session:")
    print(points_data2)

    # Convert to dictionary
    combined_points = pd.concat([points_data1, points_data2])
    points_sum = combined_points.groupby('Driver')['Points'].sum().astype(int)
    points_sum = points_sum.sort_values(ascending=False).to_dict()
    return points_sum
    
# Extract points from both sessions
race_and_sprint_points = extract_points(race_session, sprint_session)

# Save to JSON
with open(json_path, 'w') as f:
    js.dump(race_and_sprint_points, f, indent=4)
print("\n\nPoints data saved to ../data/gp_points.json\n\n")