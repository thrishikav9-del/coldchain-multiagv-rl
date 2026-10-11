"""
Cold-chain dataset loader and risk analysis.

Dataset:
    dataset/cold_chain_data.csv

The loader:
    1. Loads the CSV.
    2. Validates required columns.
    3. Cleans numeric values.
    4. Calculates a cold-chain risk score.
    5. Calculates urgency.
    6. Calculates priority.
    7. Saves the enriched dataset.
    8. Provides rows that can be consumed by the RL environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "dataset"

DATASET_PATH = DATASET_DIR / "cold_chain_data.csv"

ENRICHED_DATASET_PATH = DATASET_DIR / "cold_chain_data_enriched.csv"

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "cold_chain_data.csv"
)

ENRICHED_DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "cold_chain_data_enriched.csv"
)


# ==========================================================
# REQUIRED COLUMNS
# ==========================================================


REQUIRED_COLUMNS = [
    "timestamp",
    "vehicle_gps_latitude",
    "vehicle_gps_longitude",
    "fuel_consumption_rate",
    "eta_variation_hours",
    "traffic_congestion_level",
    "warehouse_inventory_level",
    "loading_unloading_time",
    "handling_equipment_availability",
    "order_fulfillment_status",
    "weather_condition_severity",
    "port_congestion_level",
    "shipping_costs",
    "supplier_reliability_score",
    "lead_time_days",
    "historical_demand",
    "iot_temperature",
    "cargo_condition_status",
    "route_risk_level",
    "customs_clearance_time",
    "driver_behavior_score",
    "fatigue_monitoring_score",
    "disruption_likelihood_score",
    "delay_probability",
    "risk_classification",
    "delivery_time_deviation",
]


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(
    path: str | Path | None = None,
) -> pd.DataFrame:

    if path is None:
        path = DATASET_PATH

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"""
Dataset file was not found:

{path}

Expected structure:

ColdChain_MultiAGV_RL/
├── dataset/
│   ├── cold_chain_data.csv
│   └── dataset_loader.py
"""
        )

    df = pd.read_csv(
        path
    )

    if df.empty:

        raise ValueError(
            "The dataset is empty."
        )

    # ------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Dataset is missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    # ------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    # ------------------------------------------------------
    # Numeric columns
    # ------------------------------------------------------

    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in (
            "timestamp",
            "risk_classification",
        )
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ------------------------------------------------------
    # Remove completely invalid rows
    # ------------------------------------------------------

    df = df.dropna(
        subset=[
            "iot_temperature",
            "disruption_likelihood_score",
            "delay_probability",
            "risk_classification",
        ]
    ).reset_index(
        drop=True
    )

    if df.empty:

        raise ValueError(
            "No valid rows remain after cleaning."
        )

    return df


# ==========================================================
# NORMALIZATION
# ==========================================================

def min_max_normalize(
    series: pd.Series,
) -> pd.Series:

    minimum = series.min()

    maximum = series.max()

    if not np.isfinite(minimum):

        return pd.Series(
            0.0,
            index=series.index,
        )

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        (
            series - minimum
        )
        /
        (
            maximum - minimum
        )
    ).clip(
        0.0,
        1.0,
    )


# ==========================================================
# RISK ANALYSIS
# ==========================================================

def calculate_risk_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # ------------------------------------------------------
    # Temperature risk
    #
    # Cold-chain temperature deviations are treated as risk.
    # ------------------------------------------------------

    temperature = (
        result[
            "iot_temperature"
        ].abs()
    )

    temperature_risk = (
        temperature
        /
        max(
            temperature.max(),
            1.0,
        )
    ).clip(
        0.0,
        1.0,
    )

    # ------------------------------------------------------
    # Dataset-provided risk indicators
    # ------------------------------------------------------

    disruption_risk = (
        min_max_normalize(
            result[
                "disruption_likelihood_score"
            ]
        )
    )

    delay_risk = (
        min_max_normalize(
            result[
                "delay_probability"
            ]
        )
    )

    route_risk = (
        min_max_normalize(
            result[
                "route_risk_level"
            ]
        )
    )

    weather_risk = (
        min_max_normalize(
            result[
                "weather_condition_severity"
            ]
        )
    )

    traffic_risk = (
        min_max_normalize(
            result[
                "traffic_congestion_level"
            ]
        )
    )

    delivery_deviation_risk = (
        min_max_normalize(
            result[
                "delivery_time_deviation"
            ].abs()
        )
    )

    # ------------------------------------------------------
    # Combined risk
    #
    # Higher = more dangerous.
    # ------------------------------------------------------

    result["risk_score"] = (
        0.25 * temperature_risk
        + 0.20 * disruption_risk
        + 0.20 * delay_risk
        + 0.15 * route_risk
        + 0.10 * weather_risk
        + 0.05 * traffic_risk
        + 0.05 * delivery_deviation_risk
    ).clip(
        0.0,
        1.0,
    )

    # ------------------------------------------------------
    # Urgency
    # ------------------------------------------------------

    result["urgency_score"] = (
        0.60 * result["risk_score"]
        + 0.20 * delay_risk
        + 0.20 * disruption_risk
    ).clip(
        0.0,
        1.0,
    )

    # ------------------------------------------------------
    # Priority
    # ------------------------------------------------------

    result["priority"] = pd.cut(
        result["urgency_score"],
        bins=[
            -np.inf,
            0.33,
            0.66,
            np.inf,
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
        ],
    )

    return result


# ==========================================================
# SAVE ENRICHED DATASET
# ==========================================================

def save_enriched_dataset(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """
    Save the dataset after risk_score, urgency_score,
    and priority have been calculated.
    """

    if output_path is None:

        output_path = (
            ENRICHED_DATASET_PATH
        )

    output_path = Path(
        output_path
    )

    # ------------------------------------------------------
    # Make sure output directory exists
    # ------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Save enriched dataframe
    # ------------------------------------------------------

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("ENRICHED DATASET SAVED")
    print("=" * 70)

    print(
        f"Output path : {output_path}"
    )

    print(
        f"Rows        : {len(df)}"
    )

    print(
        f"Columns     : {len(df.columns)}"
    )

    print("=" * 70)

    return output_path


# ==========================================================
# COMPLETE DATASET PIPELINE
# ==========================================================

def prepare_dataset(
    path: str | Path | None = None,
) -> pd.DataFrame:

    # ------------------------------------------------------
    # Load original dataset
    # ------------------------------------------------------

    df = load_dataset(
        path
    )

    # ------------------------------------------------------
    # Calculate risk_score,
    # urgency_score,
    # and priority
    # ------------------------------------------------------

    df = calculate_risk_scores(
        df
    )

    # ------------------------------------------------------
    # Save enriched dataset
    # ------------------------------------------------------

    save_enriched_dataset(
        df
    )

    return df


# ==========================================================
# GET MOST URGENT ROWS
# ==========================================================

def get_urgent_rows(
    df: pd.DataFrame,
    count: int = 10,
) -> pd.DataFrame:

    if "urgency_score" not in df.columns:

        df = calculate_risk_scores(
            df
        )

    return (
        df.sort_values(
            "urgency_score",
            ascending=False,
        )
        .head(
            count
        )
        .reset_index(
            drop=True
        )
    )


# ==========================================================
# MAIN TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("COLD-CHAIN DATASET LOADER")
    print("=" * 70)

    print(
        f"Dataset path:\n{DATASET_PATH}"
    )

    print(
        f"Enriched dataset path:\n"
        f"{ENRICHED_DATASET_PATH}"
    )

    # ------------------------------------------------------
    # Complete pipeline
    # ------------------------------------------------------

    df = prepare_dataset()

    print()

    print(
        f"Rows loaded      : {len(df)}"
    )

    print(
        f"Columns loaded   : {len(df.columns)}"
    )

    print()

    print(
        "Original columns:"
    )

    print(
        REQUIRED_COLUMNS
    )

    print()

    print(
        "Added columns:"
    )

    print(
        [
            "risk_score",
            "urgency_score",
            "priority",
        ]
    )

    print()

    print(
        "Risk distribution:"
    )

    print(
        df[
            "priority"
        ].value_counts()
    )

    print()

    print(
        "Top 10 urgent records:"
    )

    urgent = get_urgent_rows(
        df,
        count=10,
    )

    print(
        urgent[
            [
                "timestamp",
                "iot_temperature",
                "disruption_likelihood_score",
                "delay_probability",
                "route_risk_level",
                "risk_score",
                "urgency_score",
                "priority",
                "risk_classification",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print("=" * 70)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 70)

