# Results report and analysis

## Table of contents
- [Results report and analysis](#results-report-and-analysis)
  - [Table of contents](#table-of-contents)
  - [General information](#general-information)
    - [Round 2 - Shanghai Grand Prix Circuit, China](#round-2---shanghai-grand-prix-circuit-china)
    - [Weekend schedule](#weekend-schedule)
  - [Circuit](#circuit)
  - [Prediction](#prediction)
  - [Results](#results)
    - [Points](#points)
    - [Retirements](#retirements)
  - [Plots](#plots)
    - [Position changes](#position-changes)
    - [Tyre strategy](#tyre-strategy)
    - [Qualifying results](#qualifying-results)
    - [Team pace comparison](#team-pace-comparison)

## General information
### Round 2 - Shanghai Grand Prix Circuit, China
### Weekend schedule

## Circuit

![china_circuit](gpx/2026trackshanghaidetailed.webp)
*Source: https://www.formula1.com/en/racing/2026/china*

![china_chircuit_map](gpx/2026_china_circuit_map.png)

## Prediction
The prediction model was developed using an **XGBoost Regressor** trained on data from the 2025 Japanese Grand Prix. The target variable is the **delta between race lap times and qualifying lap times**, focusing on degradation effects from fuel, tyres, and weather. 

**Key features** include qualifying time (`QualiTime_s`), lap number (`LapNumber`), tyre life (`TyreLife`), tyre compound encoded as an ID (`Compound_Id`), and **engineered variables**: fuel burn effect (`Fuel_Burn_Effect`, linear weight reduction over laps), tyre stress (`Tyre_Stress`, exponential degradation for high-speed tracks), and track-wind interaction (`Track_Wind_Interaction`, product of track temperature and wind speed). 

Additional **weather features** such as track temperature (`TrackTemp`), air temperature (`AirTemp`), wind speed (`WindSpeed`), and wind gust (`WindGust`) were incorporated where available. 

The model was trained with parameters including **2000 estimators**, a **learning rate of 0.008**, **max depth of 5**, and regularization to prevent overfitting, achieving a mean absolute error (MAE) of 0.5050 seconds on the validation set.

| Position | Driver | Predicted Race Lap |
| -------- | ------ | ------------------ |
| 1st      | RUS    | 96.259809          |
| 2nd      | HAM    | 96.348601          |
| 3rd      | PIA    | 96.445091          |
| 4th      | LEC    | 96.457768          |
| 5th      | NOR    | 96.532887          |
| 6th      | ANT    | 96.620587          |
| 7th      | VER    | 96.772265          |
| 8th      | HAD    | 96.891265          |
| 9th      | GAS    | 96.964863          |
| 10th     | BEA    | 97.062265          |

**Model Performance:** MAE (Mean Absolute Error) = 0.5050s

## Results
### Points
| Pos | Driver                | Team         | Time / Status | Points |
| --- | --------------------- | ------------ | ------------- | ------ |
| 1   | Andrea Kimi Antonelli | Mercedes     | 1:33:15.607   | 25     |
| 2   | George Russell        | Mercedes     | +5.515s       | 18     |
| 3   | Lewis Hamilton        | Ferrari      | +25.267s      | 15     |
| 4   | Charles Leclerc       | Ferrari      | +28.894s      | 12     |
| 5   | Oliver Bearman        | Haas         | +57.268s      | 10     |
| 6   | Pierre Gasly          | Alpine       | +59.647s      | 8      |
| 7   | Liam Lawson           | Racing Bulls | +80.588s      | 6      |
| 8   | Isack Hadjar          | Red Bull     | +87.247s      | 4      |
| 9   | Carlos Sainz          | Williams     | +1 lap        | 2      |
| 10  | Franco Colapinto      | Alpine       | +1 lap        | 1      |
| 11  | Nico Hülkenberg       | Audi         | +1 lap        | 0      |
| 12  | Arvid Lindblad        | Racing Bulls | +1 lap        | 0      |
| 13  | Valtteri Bottas       | Cadillac     | +1 lap        | 0      |
| 14  | Esteban Ocon          | Haas         | +1 lap        | 0      |
| 15  | Sergio Pérez          | Cadillac     | +1 lap        | 0      |

### Retirements
| Driver            | Team         | Reason           |
| ----------------- | ------------ | ---------------- |
| Max Verstappen    | Red Bull     | DNF (Collision)  |
| Fernando Alonso   | Aston Martin | DNF              |
| Lance Stroll      | Aston Martin | DNF (Electrical) |
| Lando Norris      | McLaren      | DNS (Electrical) |
| Oscar Piastri     | McLaren      | DNS (Electrical) |
| Gabriel Bortoleto | Audi         | DNS              |
| Alexander Albon   | Williams     | DNS              |

## Plots

### Position changes
![china_position_changes](gpx/2026_china_position_changes.png)
### Tyre strategy
![china_tyre_strategy](gpx/2026_china_tyre_strategy.png)
### Qualifying results
![china_qualifying_results](gpx/2026_china_qualifying_results.png)
### Team pace comparison
![china_team_pace_comparison](gpx/2026_china_team_pace_comparison.png)