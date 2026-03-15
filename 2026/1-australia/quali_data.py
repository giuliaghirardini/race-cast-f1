import os
import fastf1
import pandas as pd

# Set up cache folder
cache_folder = 'cache_folder'
os.makedirs(cache_folder, exist_ok=True)
fastf1.Cache.enable_cache(cache_folder)

# Load session data
session = fastf1.get_session(2026, 'Australia', 'Q')
session.load(telemetry=False)
#weather_data = session.laps.get_weather_data()

print("\n\nCache enabled and data loaded successfully!\n\n")

# Extract laps and split into qualifying sessions
laps = session.laps
df = laps[['Driver', 'LapTime', 'Compound']]
qualifying_sessions = dict(zip(['Q1', 'Q2', 'Q3'], laps.split_qualifying_sessions()))

driver_numbers = {
    "NOR": 1,   # Lando Norris
    "VER": 3,   # Max Verstappen  (shared #1 as reigning champ)
    "HAD": 6,   # Isack Hadjar
    "ANT": 12,  # Andrea Kimi Antonelli
    "GAS": 10,  # Pierre Gasly
    "COL": 43,  # Franco Colapinto
    "OCO": 31,  # Esteban Ocon
    "BEA": 87,  # Oliver Bearman
    "ALO": 14,  # Fernando Alonso
    "STR": 18,  # Lance Stroll
    "SAI": 55,  # Carlos Sainz Jr.
    "ALB": 23,  # Alexander Albon
    "HUL": 27,  # Nico Hülkenberg
    "HAM": 44,  # Lewis Hamilton
    "LEC": 16,  # Charles Leclerc
    "PIA": 81,  # Oscar Piastri
    "LAW": 30,  # Liam Lawson
    "RUS": 63,  # George Russell
    "PER": 11,  # Sergio Pérez
    "BOT": 77,  # Valtteri Bottas
    "BOR": 5,   # Gabriel Bortoleto
    "LIN": 41,  # Arvid Lindblad
}

# Function to sum lap times for a given qualifying session
def best_lap_times(laps_session, driver_numbers):
    df_session = laps_session[['Driver', 'LapTime']].dropna(subset=['LapTime'])
    best_times = df_session.groupby('Driver')['LapTime'].min().reset_index()
    # Sort by best lap time before mapping numbers
    best_times = best_times.sort_values('LapTime').reset_index(drop=True)
    best_times['DriverNumber'] = best_times['Driver'].map(driver_numbers)
    return best_times[['Driver', 'DriverNumber', 'LapTime']].rename(columns={'LapTime': 'BestQualiTime'})

# Calculate and print total times for each qualifying session
for q_name, q_data in qualifying_sessions.items():
    total_times = best_lap_times(q_data, driver_numbers).sort_values('BestQualiTime')
    total_times['BestQualiTimeSeconds'] = total_times['BestQualiTime'].dt.total_seconds()        
    print(f'\n🏁 Total Qualifying Times {q_name} 🏁\n')
    total_times.index = total_times.index + 1
    print(total_times)
    
