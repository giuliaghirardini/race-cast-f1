import matplotlib.pyplot as plt
import pandas as pd
from timple.timedelta import strftimedelta

import fastf1
import fastf1.plotting
from fastf1.core import Laps
import numpy as np
import seaborn as sns
import os

######################################################
### RACE SETTINGS
######################################################

race_name = "Australia"
year = 2026
string_race = "2026_australia"

### Load session data
# Race session
race_session = fastf1.get_session(year, race_name, 'R')
race_session.load(telemetry=False, weather=False)

# Qualifying session
quali_session = fastf1.get_session(year, race_name, 'Q')
quali_session.load()

### Setting up data
laps = race_session.laps
drivers = pd.unique(quali_session.laps['Driver'])

quali_laps = quali_session.laps
pos = quali_laps.pick_fastest().get_pos_data()

circuit_info = quali_session.get_circuit_info()

### Path
path = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(path,"gpx"), exist_ok=True)
gpx_path = os.path.join(path, "gpx")

######################################################
### POSITION CHANGES DURING THE RACE
######################################################

# Load FastF1's dark color scheme
fastf1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')

# Plot
fig, ax = plt.subplots(figsize=(8.0, 4.9))

for drv in race_session.drivers:
    drv_laps = race_session.laps.pick_drivers(drv)

    if not drv_laps.empty:
        abb = drv_laps['Driver'].iloc[0]
    else:
        print("No laps found for this driver")

    style = fastf1.plotting.get_driver_style(identifier=abb,
                                             style=['color', 'linestyle'],
                                             session=race_session)

    ax.plot(drv_laps['LapNumber'], drv_laps['Position'],
            label=abb, **style)
    
ax.set_ylim([20.5, 0.5])
ax.set_yticks([1, 5, 10, 15, 20])
ax.set_xlabel('Lap')
ax.set_ylabel('Position')

ax.legend(bbox_to_anchor=(1.0, 1.02))
plt.tight_layout()
# plt.show()

# Save plot
plot_path = os.path.join(gpx_path, string_race+"_position_changes.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

######################################################
### QUALIFYING RESULTS OVERVIEW
######################################################

# Enable Matplotlib patches for plotting timedelta values
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme=None)

list_fastest_laps = list()
list_fastest_laps = []

for drv in drivers:
    drv_laps = quali_session.laps.pick_drivers(drv).pick_quicklaps()
    drv_fastest = drv_laps.pick_fastest()

    if drv_fastest is not None:
        list_fastest_laps.append(drv_fastest)

fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

pole_lap = fastest_laps.pick_fastest()
fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

team_colors = list()
for index, quali_laps in fastest_laps.iterlaps():
    color = fastf1.plotting.get_team_color(quali_laps['Team'], session=quali_session)
    team_colors.append(color)

fig, ax = plt.subplots()
ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
        color=team_colors, edgecolor='grey')
ax.set_yticks(fastest_laps.index)
ax.set_yticklabels(fastest_laps['Driver'])

# show fastest at the top
ax.invert_yaxis()

# draw vertical lines behind the bars
ax.set_axisbelow(True)
ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

plt.suptitle(f"{quali_session.event['EventName']} {quali_session.event.year} Qualifying\n"
             f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

# Save plot
plot_path = os.path.join(gpx_path, string_race+"_qualifying_results.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

######################################################
### CIRCUIT
######################################################

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

# Get an array of shape [n, 2] where n is the number of points and the second
# axis is x and y.
track = pos.loc[:, ('X', 'Y')].to_numpy()

# Convert the rotation angle from degrees to radian.
track_angle = circuit_info.rotation / 180 * np.pi

# Rotate and plot the track map.
rotated_track = rotate(track, angle=track_angle)
plt.plot(rotated_track[:, 0], rotated_track[:, 1])

offset_vector = [500, 0]  # offset length is chosen arbitrarily to 'look good'

# Iterate over all corners.
for _, corner in circuit_info.corners.iterrows():
    # Create a string from corner number and letter
    txt = f"{corner['Number']}{corner['Letter']}"

    # Convert the angle from degrees to radian.
    offset_angle = corner['Angle'] / 180 * np.pi

    # Rotate the offset vector so that it points sideways from the track.
    offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

    # Add the offset to the position of the corner
    text_x = corner['X'] + offset_x
    text_y = corner['Y'] + offset_y

    # Rotate the text position equivalently to the rest of the track map
    text_x, text_y = rotate([text_x, text_y], angle=track_angle)

    # Rotate the center of the corner equivalently to the rest of the track map
    track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

    # Draw a circle next to the track.
    plt.scatter(text_x, text_y, color='grey', s=140)

    # Draw a line from the track to this circle.
    plt.plot([track_x, text_x], [track_y, text_y], color='grey')

    # Finally, print the corner number inside the circle.
    plt.text(text_x, text_y, txt,
             va='center_baseline', ha='center', size='small', color='white')
    
plt.title(quali_session.event['Location'])
plt.axis('equal')

# Save plot
plot_path = os.path.join(gpx_path, string_race+"_circuit_map.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

######################################################
### TYRE STRATEGY DURING THE RACE
######################################################

drivers = [race_session.get_driver(driver)["Abbreviation"] for driver in race_session.drivers]

stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
stints = stints.groupby(["Driver", "Stint", "Compound"])
stints = stints.count().reset_index()

stints = stints.rename(columns={"LapNumber": "StintLength"})

fig, ax = plt.subplots(figsize=(5, 10))

for driver in race_session.drivers:
    driver_stints = stints.loc[stints["Driver"] == driver]

    previous_stint_end = 0
    for idx, row in driver_stints.iterrows():
        # each row contains the compound name and stint length
        # we can use these information to draw horizontal bars
        compound_color = fastf1.plotting.get_compound_color(row["Compound"],
                                                            session=race_session)
        plt.barh(
            y=driver,
            width=row["StintLength"],
            left=previous_stint_end,
            color=compound_color,
            edgecolor="black",
            fill=True
        )

        previous_stint_end += row["StintLength"]

plt.title(str(year)+race_name+" Grand Prix Tyre Strategies")
plt.xlabel("Lap Number")
plt.grid(False)
# invert the y-axis so drivers that finish higher are closer to the top
ax.invert_yaxis()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()

# Save plot
plot_path = os.path.join(gpx_path, string_race+"_tyre_strategy.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

##########################################################
### TEAM PACE COMPARISON
##########################################################

transformed_laps = laps.copy()
transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

# order the team from the fastest (lowest median lap time) tp slower
team_order = (
    transformed_laps[["Team", "LapTime (s)"]]
    .groupby("Team")
    .median()["LapTime (s)"]
    .sort_values()
    .index
)

# make a color palette associating team names to hex codes
team_palette = {team: fastf1.plotting.get_team_color(team, session=race_session)
                for team in team_order}

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

plt.title(str(year)+race_name+" Team Pace Comparison")
plt.grid(visible=False)

# x-label is redundant
ax.set(xlabel=None)
plt.tight_layout()

# Save plot
plot_path = os.path.join(gpx_path, string_race+"_team_pace_comparison.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')