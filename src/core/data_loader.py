from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_elements(csv_path: str | Path) -> pd.DataFrame:
    """
    Load engineering elements from CSV (validated upstream).

    Expected columns (minimum):
        - Element_ID
        - Length_m
        - Load_kN

    Optional:
        - Material
    """

    # Normalize path
    csv_path = Path(csv_path)

    # Existence check
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Minimal, conservative normalization
    # (do NOT alter values, only clean column names)
    df.columns = [c.strip() for c in df.columns]

    return df
