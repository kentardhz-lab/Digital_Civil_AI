from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.qc.runner import run_qc_packs


def run_basic_qc(
    elements_csv: Path,
    out_dir: Path,
    required_columns: Optional[List[str]] = None,
    max_missing_ratio: float = 0.0,
    enabled: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Runs the basic QC gate for the pipeline.

    Writes qc_report.json into out_dir and returns it as dict.
    """
    elements_csv = Path(elements_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(elements_csv)

    report = run_qc_packs(
        df=df,
        out_dir=out_dir,
        required_columns=required_columns,
        max_missing_ratio=max_missing_ratio,
        enabled=enabled,
    )

    return report
