import os
import fastf1
import fastf1.plotting

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fastf1.core import Laps
from timple.timedelta import strftimedelta

######################################################
### RACE SETTINGS
######################################################

race_name = "China"
year = 2026
string_race = "2026_china"

### Load session data
race_session = fastf1.get_session(year, race_name, 'R')
race_session.load(telemetry=False, weather=False)

quali_session = fastf1.get_session(year, race_name, 'Q')
quali_session.load()

### Setting up data
laps = race_session.laps
drivers = pd.unique(quali_session.laps['Driver'])

quali_laps_data = quali_session.laps
pos = quali_laps_data.pick_fastest().get_pos_data()

circuit_info = quali_session.get_circuit_info()

### Path
path = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(path, "gpx"), exist_ok=True)
gpx_path = os.path.join(path, "gpx")

######################################################
### POSITION CHANGES DURING THE RACE
######################################################

fastf1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')

fig, ax = plt.subplots(figsize=(10, 6))

for drv in race_session.drivers:
    # UPDATED: Changed pick_driver to pick_drivers
    drv_laps = race_session.laps.pick_drivers(drv)
    if not drv_laps.empty:
        abb = drv_laps['Driver'].iloc[0]
        style = fastf1.plotting.get_driver_style(identifier=abb,
                                                 style=['color', 'linestyle'],
                                                 session=race_session)
        ax.plot(drv_laps['LapNumber'], drv_laps['Position'], label=abb, **style)

ax.set_ylim([20.5, 0.5])
ax.set_yticks(range(1, 21))
ax.set_xlabel('Lap')
ax.set_ylabel('Position')
ax.set_title(f"{race_session.event['EventName']} {year} - Position Changes")

ax.legend(bbox_to_anchor=(1.0, 1.02), loc='upper left')
plt.tight_layout()

plot_path = os.path.join(gpx_path, string_race+"_position_changes.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

######################################################
### QUALIFYING RESULTS OVERVIEW
######################################################

fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme=None)

list_fastest_laps = []
for drv in drivers:
    # UPDATED: Changed pick_driver to pick_drivers
    drv_laps = quali_session.laps.pick_drivers(drv).pick_quicklaps()
    if not drv_laps.empty:
        list_fastest_laps.append(drv_laps.pick_fastest())

fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)
pole_lap = fastest_laps.pick_fastest()
fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

team_colors = [fastf1.plotting.get_team_color(lap['Team'], session=quali_session) 
               for _, lap in fastest_laps.iterlaps()]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'], color=team_colors, edgecolor='grey')
ax.set_yticks(fastest_laps.index)
ax.set_yticklabels(fastest_laps['Driver'])
ax.invert_yaxis()
ax.set_axisbelow(True)
ax.xaxis.grid(True, which='major', linestyle='--', color='black', alpha=0.3)

lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')
plt.suptitle(f"{quali_session.event['EventName']} {quali_session.event.year} Qualifying\n"
             f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

plot_path = os.path.join(gpx_path, string_race+"_qualifying_results.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

######################################################
### CIRCUIT MAP
######################################################

fig, ax = plt.subplots(figsize=(10, 10)) # Create new fig to avoid overlap

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

track = pos.loc[:, ('X', 'Y')].to_numpy()
track_angle = circuit_info.rotation / 180 * np.pi
rotated_track = rotate(track, angle=track_angle)

ax.plot(rotated_track[:, 0], rotated_track[:, 1], linestyle='-', linewidth=3)

offset_vector = [500, 0] 

for _, corner in circuit_info.corners.iterrows():
    txt = f"{corner['Number']}{corner['Letter']}"
    offset_angle = corner['Angle'] / 180 * np.pi
    offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
    
    text_x, text_y = rotate([corner['X'] + offset_x, corner['Y'] + offset_y], angle=track_angle)
    track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

    ax.scatter(text_x, text_y, color='grey', s=140)
    ax.plot([track_x, text_x], [track_y, text_y], color='grey')
    ax.text(text_x, text_y, txt, va='center_baseline', ha='center', size='small', color='white')

ax.set_title(f"{quali_session.event['Location']} Circuit Map")
ax.axis('off') # Cleaner look for maps
ax.set_aspect('equal')

plot_path = os.path.join(gpx_path, string_race+"_circuit_map.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

######################################################
### TYRE STRATEGY
######################################################

# Prepare stint data
stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
stints = stints.groupby(["Driver", "Stint", "Compound"]).count().reset_index()
stints = stints.rename(columns={"LapNumber": "StintLength"})

# Order drivers by finishing position
drivers = [race_session.get_driver(d)["Abbreviation"] for d in race_session.drivers]

fig, ax = plt.subplots(figsize=(6, 10))

for driver in drivers:
    driver_stints = stints.loc[stints["Driver"] == driver]

    previous_stint_end = 0
    for _, row in driver_stints.iterrows():
        compound_color = fastf1.plotting.get_compound_color(
            row["Compound"],
            session=race_session
        )

        ax.barh(
            y=driver,
            width=row["StintLength"],
            left=previous_stint_end,
            color=compound_color,
            edgecolor="black",
            height=0.8
        )

        previous_stint_end += row["StintLength"]

# Titles and labels
ax.set_title(f"{year} {race_name} Grand Prix Tyre Strategies")
ax.set_xlabel("Lap Number")

# Best finishing drivers at the top
ax.invert_yaxis()

# Clean aesthetics
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.grid(False)

plt.tight_layout()

# Save plot
plot_path = os.path.join(gpx_path, string_race + "_tyre_strategy.png")
fig.savefig(plot_path, dpi=300, bbox_inches="tight")

##########################################################
### TEAM PACE COMPARISON
##########################################################

transformed_laps = laps.copy()
transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

# Filter out extreme outliers for better visualization
transformed_laps = transformed_laps.loc[transformed_laps['LapTime (s)'] < transformed_laps['LapTime (s)'].min() * 1.07]

team_order = (transformed_laps[["Team", "LapTime (s)"]].groupby("Team")
              .median()["LapTime (s)"].sort_values().index)

team_palette = {team: fastf1.plotting.get_team_color(team, session=race_session) for team in team_order}

fig, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(
    data=transformed_laps,
    x="Team",
    y="LapTime (s)",
    hue="Team",
    order=team_order,
    palette=team_palette,
    whiskerprops=dict(color="white"),
    boxprops=dict(edgecolor="white"),
    medianprops=dict(color="grey"),
    capprops=dict(color="white"),
)

ax.set_title(f"{year} {race_name} Team Pace Comparison")
ax.set_xlabel(None)
plt.xticks(rotation=45)
plt.tight_layout()

plot_path = os.path.join(gpx_path, string_race+"_team_pace_comparison.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')