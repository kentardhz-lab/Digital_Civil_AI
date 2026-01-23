from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import uuid
from src.logging.logger import setup_logger
from src.core.manifest import build_run_manifest, write_manifest

from src.qc.basic_qc import run_basic_qc
from src.core.data_loader import load_elements
from src.core.decision_engine import engineering_verdict
from src.core.validation_engine import degrade_input, extreme_input


def run_full_pipeline(
    *,
    elements_csv: Path,
    run_dir: Path,
    scenarios: Dict[str, bool] | None = None,
) -> Path:
    """
    Full system orchestration (end-to-end).
    Inputs:
      - elements_csv: path to validated input CSV
      - run_dir: directory where ALL artifacts for this run will be written
      - scenarios: which scenarios to compute (base/degraded/extreme)
    Returns:
      - run_dir
    """
    scenarios = scenarios or {"base": True, "degraded": True, "extreme": True}

    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)
    run_id = uuid.uuid4().hex
    logger.info("Run started. run_id=%s", run_id)
    df = load_elements(str(elements_csv))


    # 2) QC (write report inside run_dir)
    qc_report = run_basic_qc(
        elements_csv=Path(elements_csv),
        out_dir=run_dir,
    )

    # --- QC GATE ---
    qc_status = qc_report.get("status", "PASS")

    if qc_status != "PASS":
        raise RuntimeError(
            f"QC gate failed with status={qc_status}. "
            f"See qc_report.json for details."
        )


    # 3) Compute verdicts per scenario
    cols = {}
    if scenarios.get("base", True):
        cols["Base"] = engineering_verdict(df)

    if scenarios.get("degraded", True):
        cols["Degraded"] = engineering_verdict(degrade_input(df))

    if scenarios.get("extreme", True):
        cols["Extreme"] = engineering_verdict(extreme_input(df))

    report = pd.DataFrame(cols)

    # 4) Write unified final report
    out_file = run_dir / "final_engineering_report.csv"
    report.to_csv(out_file, index=False)

    output_files = [
        run_dir / "final_engineering_report.csv",
        run_dir / "qc_report.json",
        run_dir / "pip_freeze.txt",
        run_dir / "config_used.yaml",
    ]
    manifest = build_run_manifest(
        run_id=run_id,
        config_name="demo_project.yaml",
        input_source="CSV / Excel / IFC / BIM",
        run_dir=run_dir,
        outputs=output_files,
   )
    manifest_path = write_manifest(run_dir, manifest)
    logger.info("Manifest written: %s", str(manifest_path))
    logger.info("Run finished successfully.")




    return run_dir
