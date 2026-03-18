import os
import fastf1
import fastf1.plotting

import json as js
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict

# USER SETTINGS

num_gp = 2  # scalable

################################################################
# PATH AND DATA LOADING
################################################################

# Set up for json output
path = os.path.dirname(os.path.abspath(__file__))
print(f"\nBase path is: {path}")
content = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
print(content)

# reorder content
content = sorted(content, key=lambda x: int(x.split('-')[0]))
print(f"\nReordered content: {content}")

################################################################
# TOTAL POINTS
################################################################

cumulative_points_driver = defaultdict(int)

for i in range(1, num_gp+1):
    gp_folder = content[i-1]
    print(f"\nGP folder: {gp_folder}")
    data_folder = os.path.join(path, gp_folder, "data")
    print(f"\nGP data folder: {data_folder}")
    json_namefile = 'gp_points.json'
    json_path = os.path.join(data_folder, json_namefile)
    points_data = js.load(open(json_path))
    points_per_driver = {driver: points_data[driver] for driver in points_data}

    for driver, points in points_per_driver.items():
        cumulative_points_driver[driver] += points

cumulative_points_driver = dict(sorted(cumulative_points_driver.items(), key=lambda x: x[1], reverse=True))

print("\nPoints per driver:")
for driver, points in cumulative_points_driver.items():
    print(f"  {driver}: {points}")

cumulative_file = "cumulative_points.json" 
cumulative_path = os.path.join(path, cumulative_file)
with open(cumulative_path, 'w') as f:
    js.dump(cumulative_points_driver, f, indent=4)
print(f"\nCumulative points saved to {cumulative_path}")

################################################################
# PLOT POINTS PER RACE
################################################################

# Setup FastF1 styling
fastf1.plotting.setup_mpl()

cumulative_points = defaultdict(int)
history = defaultdict(list)
gp_labels = []

# Load one session to get team info (Australia is fine)
session = fastf1.get_session(2026, "Australia", "R")
session.load()

# Map driver -> team
driver_team = {}
for _, row in session.results.iterrows():
    driver_team[row["Abbreviation"]] = row["TeamName"]

# Loop over GPs
for i in range(1, num_gp + 1):
    gp_folder = content[i-1]
    data_folder = os.path.join(path, gp_folder, "data")
    json_path = os.path.join(data_folder, "gp_points.json")

    points_data = js.load(open(json_path))
    gp_labels.append(gp_folder)

    # Update cumulative
    for driver, pts in points_data.items():
        cumulative_points[driver] += pts

    # Store history
    for driver in cumulative_points:
        history[driver].append(cumulative_points[driver])

    # Fill missing drivers
    for driver in history:
        if len(history[driver]) < i:
            history[driver].append(history[driver][-1])

# Determine team order (leader vs teammate)
team_drivers = defaultdict(list)
for driver, team in driver_team.items():
    if driver in history:
        team_drivers[team].append(driver)

# Sort drivers inside each team by final points
driver_style = {}
for team, drivers in team_drivers.items():
    drivers_sorted = sorted(drivers, key=lambda d: history[d][-1], reverse=True)

    for i, d in enumerate(drivers_sorted):
        driver_style[d] = '-' if i == 0 else '--'  # leader solid, teammate dashed

# Sort overall drivers by points
sorted_drivers = sorted(history.items(), key=lambda x: x[1][-1], reverse=True)

# Plot
plt.figure(figsize=(10, 6))

for driver, points in sorted_drivers:
    team = driver_team.get(driver, None)

    color = fastf1.plotting.get_team_color(team, session=session) if team else "white"
    linestyle = driver_style.get(driver, '-')

    plt.plot(
        gp_labels,
        points,
        marker='o',
        label=driver,
        color=color,
        linestyle=linestyle,
        linewidth=2
    )

plt.title(f"2026 Drivers Points (After {num_gp} Races)")
plt.xlabel("Grand Prix")
plt.ylabel("Cumulative Points")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()

save_path = os.path.join(path, "points_progression.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nPoints progression plot saved to {save_path}")

################################################################
# PLOT POINTS THERMAL MAP
################################################################

gp_labels = []
all_points = {}

# Load data
for i in range(1, num_gp + 1):
    gp_folder = content[i-1]
    data_folder = os.path.join(path, gp_folder, "data")
    json_path = os.path.join(data_folder, "gp_points.json")

    with open(json_path) as f:
        points_data = js.load(f)

    gp_labels.append(gp_folder)

    for driver, pts in points_data.items():
        all_points.setdefault(driver, []).append(pts)

# Fill missing races with 0
for driver in all_points:
    if len(all_points[driver]) < num_gp:
        all_points[driver] += [0] * (num_gp - len(all_points[driver]))

# Create DataFrame
df = pd.DataFrame(all_points, index=gp_labels).T

# Keep top 10 drivers
df["Total"] = df.sum(axis=1)
df = df.sort_values("Total", ascending=False).head(22)
df = df.drop(columns="Total")

# Plot heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(
    df,
    annot=True,
    fmt=".0f",
    cmap="cividis",        # better contrast than viridis
    linewidths=0.5,
    cbar_kws={"label": "Points"}
)

plt.title(f"2026 Points Heatmap (After {num_gp} Races)")
plt.xlabel("Grand Prix")
plt.ylabel("Driver")

plt.tight_layout()

save_path = os.path.join(path, "points_thermal_map.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nPoints thermal map saved to {save_path}")

################################################################
# PLOT CONSTRUCTOR POINTS
################################################################

# --- Map driver -> team using your existing session ---
driver_team = {}
for _, row in session.results.iterrows():
    driver_team[row["Abbreviation"]] = row["TeamName"]

# --- Aggregate constructor points ---
constructors_points = defaultdict(float)

for i in range(1, num_gp + 1):
    gp_folder = content[i-1]
    data_folder = os.path.join(path, gp_folder, "data")
    json_path = os.path.join(data_folder, "gp_points.json")

    with open(json_path) as f:
        points_data = js.load(f)

    for driver, pts in points_data.items():
        team = driver_team.get(driver)

        if team:  # safety check
            constructors_points[team] += pts

# --- Convert to sorted lists ---
teams = list(constructors_points.keys())
points = list(constructors_points.values())

# Sort by points descending
teams, points = zip(*sorted(zip(teams, points), key=lambda x: x[1], reverse=True))

# --- Colors (FastF1 style) ---
colors = [
    fastf1.plotting.get_team_color(team, session=session)
    for team in teams
]

# --- Plot ---
plt.figure(figsize=(10, 6))

plt.bar(teams, points, color=colors, edgecolor="black")

plt.title(f"2026 Constructor Standings (After {num_gp} Races)")
plt.ylabel("Points")
plt.xlabel("Team")

plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

save_path = os.path.join(path, "constructor_points.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nConstructor points plot saved to {save_path}")