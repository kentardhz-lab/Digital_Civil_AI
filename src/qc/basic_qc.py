# src/qc/basic_qc.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.qc.runner import run_qc_packs


def run_basic_qc(elements_csv: Path, out_dir: Path) -> dict:
    """
    Wrapper QC entrypoint used by the pipeline.
    Reads elements CSV, runs enabled QC packs via runner, writes qc_report.json into out_dir,
    and returns the qc_report as a dict.
    """
    df = pd.read_csv(elements_csv)

    # IMPORTANT: این لیست باید دقیقاً با ستون‌های elements.csv تو یکی باشد
    required = ["Element", "Length_m", "Load_kN"]

    report = run_qc_packs(
        df=df,
        out_dir=out_dir,
        required_columns=required,
        max_missing_ratio=0.0,
        enabled={"missing_values": True},
    )

    return report
