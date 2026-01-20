import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/elements.csv")

def load_elements():
    return pd.read_csv(DATA_PATH)
