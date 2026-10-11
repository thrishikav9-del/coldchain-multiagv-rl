from pathlib import Path

import numpy as np
import pandas as pd


class ColdChainPredictor:

    def __init__(self, dataset_path=None):

        if dataset_path is None:
            dataset_path = (
                Path(__file__).resolve().parent
                / "dataset"
                / "cold_chain_data.csv"
            )

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        self.df = pd.read_csv(
            self.dataset_path
        )

        if self.df.empty:
            raise ValueError(
                "Dataset is empty."
            )

        self.required_columns = [
            "timestamp",
            "iot_temperature",
            "weather_condition_severity",
            "route_risk_level",
            "disruption_likelihood_score",
            "delay_probability",
            "risk_classification",
            "delivery_time_deviation",
        ]

        missing = [
            column
            for column in self.required_columns
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                "Missing dataset columns: "
                + str(missing)
            )

        self.df["timestamp"] = pd.to_datetime(
            self.df["timestamp"],
            errors="coerce",
        )

        self.df = self.df.dropna(
            subset=[
                "timestamp",
                "iot_temperature",
                "weather_condition_severity",
                "route_risk_level",
                "disruption_likelihood_score",
                "delay_probability",
                "delivery_time_deviation",
            ]
        ).reset_index(drop=True)

    # ==================================================
    # CURRENT CONDITION
    # ==================================================

    def get_current_condition(
        self,
        index=0,
    ):

        index = int(index)

        index = max(
            0,
            min(
                index,
                len(self.df) - 1,
            ),
        )

        row = self.df.iloc[index]

        return {
            "timestamp":
                row["timestamp"],

            "temperature":
                float(
                    row["iot_temperature"]
                ),

            "weather_severity":
                float(
                    row[
                        "weather_condition_severity"
                    ]
                ),

            "route_risk":
                float(
                    row[
                        "route_risk_level"
                    ]
                ),

            "disruption_probability":
                float(
                    row[
                        "disruption_likelihood_score"
                    ]
                ),

            "delay_probability":
                float(
                    row[
                        "delay_probability"
                    ]
                ),

            "delivery_time_deviation":
                float(
                    row[
                        "delivery_time_deviation"
                    ]
                ),

            "risk_classification":
                str(
                    row[
                        "risk_classification"
                    ]
                ),
        }

    # ==================================================
    # RISK SCORE
    # ==================================================

    def calculate_risk_score(
        self,
        index=0,
    ):

        condition = (
            self.get_current_condition(
                index
            )
        )

        temperature = abs(
            condition["temperature"]
        )

        weather = (
            condition["weather_severity"]
        )

        route_risk = (
            condition["route_risk"]
        )

        disruption = (
            condition[
                "disruption_probability"
            ]
        )

        delay = (
            condition[
                "delay_probability"
            ]
        )

        temperature_score = min(
            temperature / 10.0,
            1.0,
        )

        weather_score = min(
            weather / 10.0,
            1.0,
        )

        route_score = min(
            route_risk / 10.0,
            1.0,
        )

        risk_score = (
            0.20 * temperature_score
            +
            0.20 * weather_score
            +
            0.20 * route_score
            +
            0.20 * disruption
            +
            0.20 * delay
        )

        return float(
            np.clip(
                risk_score,
                0.0,
                1.0,
            )
        )

    # ==================================================
    # RISK LEVEL
    # ==================================================

    def get_risk_level(
        self,
        index=0,
    ):

        score = (
            self.calculate_risk_score(
                index
            )
        )

        if score >= 0.75:
            return "CRITICAL"

        if score >= 0.50:
            return "HIGH"

        if score >= 0.25:
            return "MODERATE"

        return "LOW"

    # ==================================================
    # FUTURE RISK
    # ==================================================

    def predict_future_risk(
        self,
        index=0,
        horizon=5,
    ):

        index = int(index)
        horizon = max(
            1,
            int(horizon),
        )

        end = min(
            index + horizon,
            len(self.df),
        )

        if index >= end:
            return {
                "current_risk": 0.0,
                "future_risk": 0.0,
                "risk_increasing": False,
            }

        current_risk = (
            self.calculate_risk_score(
                index
            )
        )

        future_scores = []

        for i in range(
            index,
            end,
        ):

            future_scores.append(
                self.calculate_risk_score(
                    i
                )
            )

        future_risk = float(
            np.mean(
                future_scores
            )
        )

        return {
            "current_risk":
                current_risk,

            "future_risk":
                future_risk,

            "risk_increasing":
                future_risk
                >
                current_risk,
        }

    # ==================================================
    # DATASET SUMMARY
    # ==================================================

    def summary(self):

        return {
            "rows":
                len(self.df),

            "columns":
                len(self.df.columns),

            "temperature_min":
                float(
                    self.df[
                        "iot_temperature"
                    ].min()
                ),

            "temperature_max":
                float(
                    self.df[
                        "iot_temperature"
                    ].max()
                ),

            "temperature_mean":
                float(
                    self.df[
                        "iot_temperature"
                    ].mean()
                ),

            "risk_distribution":
                self.df[
                    "risk_classification"
                ]
                .value_counts()
                .to_dict(),
        }


if __name__ == "__main__":

    predictor = (
        ColdChainPredictor()
    )

    print(
        "\n========================================"
    )
    print(
        "COLD CHAIN DATASET"
    )
    print(
        "========================================"
    )

    print(
        "Rows:",
        len(predictor.df),
    )

    print(
        "Columns:",
        len(predictor.df.columns),
    )

    print(
        "\nFirst condition:"
    )

    print(
        predictor.get_current_condition(
            0
        )
    )

    print(
        "\nRisk score:"
    )

    print(
        predictor.calculate_risk_score(
            0
        )
    )

    print(
        "\nRisk level:"
    )

    print(
        predictor.get_risk_level(
            0
        )
    )

    print(
        "\nFuture risk:"
    )

    print(
        predictor.predict_future_risk(
            0,
            horizon=5,
        )
    )

    print(
        "\nSummary:"
    )

    print(
        predictor.summary()
    )
    