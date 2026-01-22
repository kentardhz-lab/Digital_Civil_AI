from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["Element_ID", "Length_m", "Load_kN"]

def run_basic_qc(elements_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(elements_csv)

    report = {
        "missing_columns": [],
        "missing_values": {},
        "row_count": int(len(df)),
    }

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            report["missing_columns"].append(col)
        else:
            missing = int(df[col].isna().sum())
            if missing > 0:
                report["missing_values"][col] = missing

    out_dir.mkdir(parents=True, exist_ok=True)
    qc_path = out_dir / "qc_report.json"
    qc_path.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
