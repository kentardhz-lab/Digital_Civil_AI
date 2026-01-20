from ..core.data_loader import load_elements
from ..core.decision_engine import engineering_verdict
from ..core.validation_engine import degrade_input, extreme_input

import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

def run():
    df = load_elements()

    report = pd.DataFrame({
        "Base": engineering_verdict(df),
        "Degraded": engineering_verdict(degrade_input(df)),
        "Extreme": engineering_verdict(extreme_input(df))
    })

    report.to_csv(OUTPUT_PATH / "final_engineering_report.csv", index=False)

if __name__ == "__main__":
    run()
