import pandas as pd

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

# 5) Validation
if (df["Load_per_meter"] > 50).any():
    print("WARNING: Unusually high load per meter detected")

print("\n--- FINAL DATA ---")
print(df)