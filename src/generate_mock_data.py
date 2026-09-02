from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data"
data_dir.mkdir(exist_ok=True)

np.random.seed(42)
dates = pd.date_range(start="2026-01-01", periods=90, freq="D")
players = [f"P{i:03d}" for i in range(1, 20)]

rows = []
for player in players:
    base_fitness = np.random.normal(1.0, 0.15)
    for date in dates:
        total_dist = max(
            2000, round(np.random.normal(8500 * base_fitness, 1500))
        )
        hsr = max(100, round(np.random.normal(900 * base_fitness, 300)))
        sprint = max(20, round(np.random.normal(250 * base_fitness, 100)))

        rows.append(
            {
                "Player_ID": player,
                "Date": date.strftime("%Y-%m-%d"),
                "Total_Distance_m": total_dist,
                "High_Speed_Running_m": hsr,
                "Sprint_Distance_m": sprint,
            }
        )

df = pd.DataFrame(rows)
df = df.sort_values(by=["Player_ID", "Date"]).reset_index(drop=True)

# Feature construction for target injection
df["Daily_Load"] = (
    df["Total_Distance_m"] * 0.1
    + df["High_Speed_Running_m"] * 0.5
    + df["Sprint_Distance_m"] * 1.0
)
df["Acute_Load"] = df.groupby("Player_ID")["Daily_Load"].transform(
    lambda x: x.ewm(span=7, min_periods=3).mean()
)
df["Chronic_Load"] = df.groupby("Player_ID")["Daily_Load"].transform(
    lambda x: x.ewm(span=28, min_periods=14).mean()
)
df["ACWR"] = df["Acute_Load"] / (df["Chronic_Load"] + 1e-5)

# Sports science ground truth logic: High ACWR (>1.5) or extreme sprinting (>400m) raises injury probability
base_p = 0.01
acwr_risk = np.where(
    df["ACWR"] > 1.5, 0.30, np.where(df["ACWR"] > 1.3, 0.10, 0.0)
)
sprint_risk = np.where(df["Sprint_Distance_m"] > 400, 0.20, 0.0)
p_injury = np.clip(base_p + acwr_risk + sprint_risk, 0, 1)

df["Injury_Occurred"] = np.random.binomial(1, p_injury)

# Export clean raw fact table
export_cols = [
    "Player_ID",
    "Date",
    "Total_Distance_m",
    "High_Speed_Running_m",
    "Sprint_Distance_m",
    "Injury_Occurred",
]
df[export_cols].to_csv(data_dir / "Fact_Player_Workload.csv", index=False)
print("Updated 'data/Fact_Player_Workload.csv' with domain relationship logic.")