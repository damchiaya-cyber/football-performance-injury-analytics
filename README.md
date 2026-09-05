# Football Performance & Injury Analytics

A practical data pipeline and machine learning project built to predict non-contact soft-tissue injury risk in professional football players using daily workload data.

## Why This Project?
Non-contact soft-tissue injuries usually happen when a player's training workload spikes too quickly without enough baseline fitness to back it up. While sports performance teams collect massive amounts of daily GPS/telemetry data, turning those raw numbers into clear, early warning signs before matchday is a challenge.

This project automates that workflow: taking raw session logs, computing key sports science metrics, and scoring player risk using machine learning so coaches can adjust training loads before an injury occurs.

## How It Works
* **Data Modeling:** Ingests daily telemetry into a relational Star Schema (`Dim_Players`, `Dim_Matches`, `Fact_Player_Workload_Scored`) for fast aggregation.
* **Feature Engineering:** Calculates rolling workload metrics automatically:
  * **Acute Load:** 7-day average short-term workload (fatigue).
  * **Chronic Load:** 28-day average long-term workload (fitness base).
  * **ACWR:** Ratio comparing acute fatigue against chronic fitness ($ACWR = \frac{\text{Acute Load}}{\text{Chronic Load}}$).
  * **Monotony & Strain:** Day-to-day workload variance and overall physiological stress.
* **Risk Prediction:** A Random Forest model classifies players into risk categories and outputs continuous injury probabilities (achieving a **0.722 ROC-AUC**).
* **Operational Dashboard:** Feeds the model output into a Power BI report (**Squad Readiness & Workload Risk Intelligence**) with dynamic date filters and visual risk matrix tools.

## Tech Stack
* **Language & Analysis:** Python (Pandas, NumPy, Scikit-Learn)
* **Data Architecture:** SQL Star Schema design
* **Business Intelligence:** Power BI

## Quick Start

```bash
# Clone the repository
git clone [https://github.com/damchiaya-cyber/football-performance-injury-analytics.git](https://github.com/damchiaya-cyber/football-performance-injury-analytics.git)
cd football-performance-injury-analytics

# Install dependencies
py -m pip install -r requirements.txt

# Run the pipeline script
py src/generate_dimensions.py