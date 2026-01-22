from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.qc.basic_qc import run_basic_qc
from src.core.data_loader import load_elements
from src.core.decision_engine import engineering_verdict
from src.core.validation_engine import degrade_input, extreme_input


# ----------------------------
# OUTPUT DIRECTORY RESOLUTION
# ----------------------------
RUN_DIR = os.environ.get("CIVIL_AI_RUN_DIR")
if RUN_DIR:
    OUTPUT_DIR = Path(RUN_DIR)
else:
    OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run() -> Path:
    # 1) Load data
    df = load_elements()

    # 2) QC (writes qc_report.json into OUTPUT_DIR)
    run_basic_qc(
        elements_csv=Path("data/elements.csv"),
        out_dir=OUTPUT_DIR,
    )

    # 3) Engineering verdicts
    report = pd.DataFrame(
        {
            "Base": engineering_verdict(df),
            "Degraded": engineering_verdict(degrade_input(df)),
            "Extreme": engineering_verdict(extreme_input(df)),
        }
    )

    # 4) Export unified report into OUTPUT_DIR
    out_file = OUTPUT_DIR / "final_engineering_report.csv"
    report.to_csv(out_file, index=False)

    return OUTPUT_DIR


if __name__ == "__main__":
    out_dir = run()
    print(f"[OK] Report written to: {out_dir / 'final_engineering_report.csv'}")
    print(f"[OK] Outputs at: {out_dir}")

