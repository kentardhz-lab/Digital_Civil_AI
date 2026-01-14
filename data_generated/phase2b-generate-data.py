import pandas as pd
import numpy as np


def generate_synthetic_rows(elements, n_per_element=30, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    # Engineering-inspired ranges and load rules (simple + explainable)
    # You can refine these later; for now we want MORE data but still structured.
    rules = {
        "Column": {"len_min": 2.5, "len_max": 4.5, "base": 120, "slope": -8, "noise": 10},
        "Beam":   {"len_min": 3.0, "len_max": 8.0, "base": 90,  "slope": -6, "noise": 10},
        "Slab":   {"len_min": 2.0, "len_max": 6.0, "base": 70,  "slope": -4, "noise": 8},
    }

    for el in elements:
        if el not in rules:
            # Default fallback for unknown elements
            rule = {"len_min": 2.0, "len_max": 8.0, "base": 80, "slope": -5, "noise": 10}
        else:
            rule = rules[el]

        lengths = rng.uniform(rule["len_min"], rule["len_max"], size=n_per_element)

        # Load rule: base + slope * (length - reference) + noise
        # Reference chosen as midpoint to avoid extreme values
        ref = (rule["len_min"] + rule["len_max"]) / 2
        noise = rng.normal(0, rule["noise"], size=n_per_element)

        loads = rule["base"] + rule["slope"] * (lengths - ref) + noise

        # Clamp to non-negative (defensive)
        loads = np.clip(loads, 1, None)

        for i in range(n_per_element):
            rows.append(
                {
                    "Element": el,
                    "Length_m": round(float(lengths[i]), 2),
                    "Load_kN": round(float(loads[i]), 1),
                }
            )

    return pd.DataFrame(rows)


def main():
    df_real = pd.read_csv("data/elements.csv")

    # Determine which elements exist in real data
    elements = sorted(df_real["Element"].dropna().unique().tolist())

    df_syn = generate_synthetic_rows(elements, n_per_element=40, seed=42)

    # Combine: keep real + synthetic
    df_combined = pd.concat([df_real, df_syn], ignore_index=True)

    # Save
    df_syn.to_csv("data_generated/elements_synthetic.csv", index=False)
    df_combined.to_csv("data_generated/elements_combined.csv", index=False)

    print("=== Phase 2.3: Data Engineering ===")
    print(f"Real rows      : {len(df_real)}")
    print(f"Synthetic rows : {len(df_syn)}")
    print(f"Combined rows  : {len(df_combined)}")
    print("\nSaved:")
    print(" - data_generated/elements_synthetic.csv")
    print(" - data_generated/elements_combined.csv")


if __name__ == "__main__":
    main()
