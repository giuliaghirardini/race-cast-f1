import os
import fastf1
import pandas as pd

# Set up cache folder one level up from current directory
cache_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), '__cache_folder')
os.makedirs(cache_folder, exist_ok=True)
fastf1.Cache.enable_cache(cache_folder)

# Load session data
session = fastf1.get_session(2025, 'Japan', 'Q')
session.load(telemetry=False)
weather_data = session.laps.get_weather_data()

print("\n\nCache enabled and data loaded successfully!\n\n")

# Extract laps and split into qualifying sessions
laps = session.laps
df = laps[['Driver', 'LapTime', 'Compound']]
qualifying_sessions = dict(zip(['Q1', 'Q2', 'Q3'], laps.split_qualifying_sessions()))

driver_numbers = {
        "ALB": 23,  # Alexander Albon
        "ALO": 14,  # Fernando Alonso
        "ANT": 7,   # Andrea Kimi Antonelli
        "BEA": 87,  # Oliver Bearman
        "BOR": 10,  # Gabriel Bortoleto
        "BOT": 77,  # Valtteri Bottas
        "COL": 43,  # Franco Colapinto
        "DEV": 45,  # Nyck de Vries
        "DOO": 61,  # Jack Doohan
        "GAS": 10,  # Pierre Gasly
        "HAD": 20,  # Isack Hadjar
        "HAM": 44,  # Lewis Hamilton
        "HUL": 27,  # Nico Hülkenberg
        "LAW": 30,  # Liam Lawson
        "LEC": 16,  # Charles Leclerc
        "MAG": 20,  # Kevin Magnussen
        "NOR": 4,   # Lando Norris
        "OCO": 31,  # Esteban Ocon
        "PER": 11,  # Sergio Pérez
        "PIA": 81,  # Oscar Piastri
        "RIC": 3,   # Daniel Ricciardo
        "RUS": 63,  # George Russell
        "SAI": 55,  # Carlos Sainz Jr.
        "SAR": 2,   # Logan Sargeant
        "STR": 18,  # Lance Stroll
        "TSU": 22,  # Yuki Tsunoda
        "VER": 1,   # Max Verstappen
        "ZHO": 24   # Zhou Guanyu
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

qualifying_2025 = pd.DataFrame(total_times[["Driver", "BestQualiTimeSeconds"]])
qualifying_2025.rename(columns={"BestQualiTimeSeconds": "QualifyingTime (s)"}, inplace=True)
print(qualifying_2025)