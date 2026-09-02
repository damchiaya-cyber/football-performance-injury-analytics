import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def load_and_engineer_features(file_path: str) -> pd.DataFrame:
    """Loads workload data and computes dynamic fatigue metrics per player."""
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Player_ID", "Date"]).reset_index(drop=True)

    # 1. Total Daily Load Composite
    df["Daily_Load"] = (
        df["Total_Distance_m"] * 0.1
        + df["High_Speed_Running_m"] * 0.5
        + df["Sprint_Distance_m"] * 1.0
    )

    # 2. EWMA Acute (7d) & Chronic (28d) Workload
    df["Acute_Load"] = df.groupby("Player_ID")["Daily_Load"].transform(
        lambda x: x.ewm(span=7, min_periods=3).mean()
    )
    df["Chronic_Load"] = df.groupby("Player_ID")["Daily_Load"].transform(
        lambda x: x.ewm(span=28, min_periods=14).mean()
    )

    # 3. ACWR Calculation
    df["ACWR"] = df["Acute_Load"] / (df["Chronic_Load"] + 1e-5)

    # 4. Monotony & Strain Metrics
    rolling_std = df.groupby("Player_ID")["Daily_Load"].transform(
        lambda x: x.rolling(7, min_periods=3).std()
    )
    df["Monotony"] = df["Acute_Load"] / (rolling_std + 1e-5)
    df["Strain"] = df["Acute_Load"] * df["Monotony"]

    return df.fillna(0)


def train_and_score():
    input_path = "data/Fact_Player_Workload.csv"
    output_path = "data/Fact_Player_Workload_Scored.csv"

    df = load_and_engineer_features(input_path)

    # Target variable check (assumes 'Injury_Occurred' binary flag in Fact table)
    features = [
        "ACWR",
        "Acute_Load",
        "Chronic_Load",
        "High_Speed_Running_m",
        "Sprint_Distance_m",
        "Monotony",
        "Strain",
    ]
    X = df[features]
    y = df["Injury_Occurred"]

    # Temporal split preferred for sports telemetry
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train interpretable Random Forest
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print("--- Model Performance ---")
    print(classification_report(y_test, y_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.3f}\n")

    # Feature Importance Analysis
    importances = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print("--- Feature Importances ---")
    print(importances.round(4))

    # Score full dataset for Power BI ingestion
    df["Injury_Risk_Probability"] = model.predict_proba(X)[:, 1].round(4)
    df["Risk_Level"] = pd.cut(
        df["Injury_Risk_Probability"],
        bins=[-0.1, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"],
    )

    df.to_csv(output_path, index=False)
    print(f"\nScored dataset successfully saved to: {output_path}")


if __name__ == "__main__":
    train_and_score()