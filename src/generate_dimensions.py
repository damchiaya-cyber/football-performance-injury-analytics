from pathlib import Path
import numpy as np
import pandas as pd

# Setup path to the data folder
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data"

# 1. Create Dim_Players
np.random.seed(42)
player_ids = [f"P{i:03d}" for i in range(1, 20)]
positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

players_data = []
for pid in player_ids:
    players_data.append({
        "Player_ID": pid,
        "Player_Name": f"Athlete {pid}",
        "Position": np.random.choice(positions, p=[0.1, 0.35, 0.35, 0.2]),
        "Age": np.random.randint(18, 35),
        "Weight_kg": np.random.randint(70, 95)
    })

df_players = pd.DataFrame(players_data)
df_players.to_csv(data_dir / "Dim_Players.csv", index=False)
print("✅ Dim_Players.csv created successfully.")

# 2. Create Dim_Matches (Calendar mapping 90 days)
dates = pd.date_range(start="2026-01-01", periods=90, freq="D")

matches_data = []
for d in dates:
    # Assign Saturdays as Matchdays, everything else as Training
    is_match = 1 if d.weekday() == 5 else 0 
    session_type = "League Match" if is_match else "Training"
    
    matches_data.append({
        "Date": d.strftime("%Y-%m-%d"),
        "Day_of_Week": d.strftime("%A"),
        "Session_Type": session_type,
        "Is_Matchday": is_match
    })

df_matches = pd.DataFrame(matches_data)
df_matches.to_csv(data_dir / "Dim_Matches.csv", index=False)
print("✅ Dim_Matches.csv created successfully.")