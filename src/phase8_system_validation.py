import pandas as pd
import numpy as np

from pathlib import Path

DATA_PATH = Path("data/elements.csv")
OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)


def load_data():
    return pd.read_csv(DATA_PATH)


def input_degradation_test(df):
    degraded = df.copy()
    degraded["Load_kN"] *= 1.25
    degraded["Length_m"] *= 0.9
    return degraded


def extreme_scenario_test(df):
    extreme = df.copy()
    extreme["Load_kN"] *= 1.6
    extreme["Length_m"] *= 0.7
    return extreme


def decision_stub(df):
    load_per_meter = df["Load_kN"] / df["Length_m"]

    verdict = np.where(
        load_per_meter < 25, "Safe",
        np.where(load_per_meter < 40, "Borderline", "Risky")
    )

    return verdict


def run_validation():
    df = load_data()

    base_verdict = decision_stub(df)
    degraded_verdict = decision_stub(input_degradation_test(df))
    extreme_verdict = decision_stub(extreme_scenario_test(df))

    report = pd.DataFrame({
        "Base": base_verdict,
        "Degraded": degraded_verdict,
        "Extreme": extreme_verdict
    })

    report.to_csv(OUTPUT_PATH / "phase8_validation_report.csv", index=False)


if __name__ == "__main__":
    run_validation()
