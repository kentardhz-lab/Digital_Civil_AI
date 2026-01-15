import pandas as pd
import matplotlib.pyplot as plt
import os

print("Phase 1 – Digital Civil Engineering")

# 1) Read data
df = pd.read_csv("data/elements.csv")
print("\n--- RAW DATA ---")
print(df)

# 2) Data Check
print("\n--- DF INFO (RAW) ---")
print(df.info())

print("\n--- MISSING VALUES (RAW) ---")
print(df.isnull().sum())

# 3) Cleaning (قبل از محاسبه)
df = df.dropna()
df = df[df["Length_m"] > 0]
df = df[df["Load_kN"] > 0]

print("\n--- CLEANED DATA ---")
print(df)

# 4) Engineering calculations
df["Load_per_meter"] = df["Load_kN"] / df["Length_m"]

print("\nTotal Load:", df["Load_kN"].sum())
print("Average Length:", df["Length_m"].mean())

# --- OUTPUTS (Phase 1 Visualization) ---
os.makedirs("outputs", exist_ok=True)

# Histogram: Load per meter
plt.figure()
df["Load_per_meter"].hist(bins=10)
plt.title("Load per meter distribution")
plt.xlabel("kN/m")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/load_per_meter_hist.png", dpi=150)
plt.close()

# Scatter: Length vs Load per meter
plt.figure()
plt.scatter(df["Length_m"], df["Load_per_meter"])
plt.title("Load per meter vs Length")
plt.xlabel("Length (m)")
plt.ylabel("Load per meter (kN/m)")
plt.tight_layout()
plt.savefig("outputs/load_per_meter_vs_length.png", dpi=150)
plt.close()

print("Phase 1 plots saved to outputs/")

# 5) Validation
if (df["Load_per_meter"] > 50).any():
    print("WARNING: Unusually high load per meter detected")

print("\n--- FINAL DATA ---")
print(df)