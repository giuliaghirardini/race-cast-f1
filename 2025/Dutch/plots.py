import fastf1
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emoji

# Enable FastF1 caching
cache_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), '__cache_folder')
os.makedirs(cache_folder, exist_ok=True)
fastf1.Cache.enable_cache(cache_folder)

# Load FastF1 2025 Dutch GP race session
session = fastf1.get_session(2025, 'Dutch', 'R')
session.load(telemetry=True)

laps = session.laps
total_laps = session.total_laps
num_drivers = len(session.drivers)

positions_per_lap = laps[["Driver", "LapNumber", "Position"]].copy()

# Calculate the number of laps completed by each driver
laps_completed = positions_per_lap.groupby("Driver")["LapNumber"].max().reset_index()
laps_completed = laps_completed.sort_values("LapNumber")
print(laps_completed)

# Order drivers by laps completed (those with fewer laps at the bottom of the legend)
ordered_drivers = laps_completed["Driver"].tolist()
print(ordered_drivers)

# Map drivers to teams
driver_team_map = {}
for drv in ordered_drivers:
    drv_team = laps[laps['Driver'] == drv]['Team'].iloc[0]
    driver_team_map[drv] = drv_team

# Assign a unique color to each team
teams = list({team for team in driver_team_map.values()})
team_colors = plt.colormaps.get_cmap('tab20', len(teams))
team_color_map = {team: team_colors(i) for i, team in enumerate(teams)}

# Find both drivers for each team and assign line style
team_drivers = {}
for drv, team in driver_team_map.items():
    team_drivers.setdefault(team, []).append(drv)

linestyles = ['-', '--']

plt.figure(figsize=(12, 8))
for team, drivers in team_drivers.items():
    for idx, drv in enumerate(drivers):
        drv_laps = positions_per_lap[positions_per_lap["Driver"] == drv]
        color = team_color_map[team]
        linestyle = linestyles[idx % 2]  # '-' for first, '--' for second
        plt.plot(
            drv_laps["LapNumber"],
            drv_laps["Position"],
            label=f"{drv} ({team})",
            color=color,
            linestyle=linestyle
        )

plt.xlabel("Lap Number")
plt.ylabel("Position")
plt.title("Driver Positions per Lap - Dutch GP 2025")
plt.gca().invert_yaxis()  # Lower position numbers are better
plt.legend(title="Driver (Team)", bbox_to_anchor=(1.05, 1), loc='upper left')