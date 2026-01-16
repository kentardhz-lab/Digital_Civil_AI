# Phase 3 — Feature Engineering & Model Comparison
import pandas as pd

# Load combined dataset (same logic as Phase 2)
df = pd.read_csv("data/elements.csv")

# Feature 1: load per unit length
df["load_per_meter"] = df["Load_kN"] / df["Length_m"]

print(df[["Load_kN", "Length_m", "load_per_meter"]].head())
print(df["load_per_meter"].describe())
print("NaN count:", df["load_per_meter"].isna().sum())
print("Inf count:", (df["load_per_meter"] == float("inf")).sum())
df["inv_length"] = 1 / df["Length_m"]
print(df[["Length_m", "inv_length"]].head())
print("NaN count (inv_length):", df["inv_length"].isna().sum())
print("Inf count (inv_length):", (df["inv_length"] == float("inf")).sum())