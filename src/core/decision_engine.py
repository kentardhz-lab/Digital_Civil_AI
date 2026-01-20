import numpy as np

def engineering_verdict(df):
    load_per_meter = df["Load_kN"] / df["Length_m"]

    return np.where(
        load_per_meter < 25, "Safe",
        np.where(load_per_meter < 40, "Borderline", "Risky")
    )
