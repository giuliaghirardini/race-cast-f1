import os
import fastf1
import json as js

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
session = fastf1.get_session(year, race_name, 'R')
session.load(telemetry=False)
print("\n-------------------------------------------------------------------\n")

print("\nData loaded successfully!\n")

################################################################
# DATA EXTRACTION AND PROCESSING
################################################################

# Extract points for each driver
points_data = session.results[['Abbreviation', 'Points']].dropna(subset=['Points'])
points_data = points_data.rename(columns={'Abbreviation': 'Driver'})
points_data['Points'] = points_data['Points'].astype(int)

# Convert to dictionary
points_dict = points_data.set_index('Driver')['Points'].to_dict()

# Save to JSON
with open(json_path, 'w') as f:
    js.dump(points_dict, f, indent=4)
print("\n\nPoints data saved to ../data/gp_points.json\n\n")