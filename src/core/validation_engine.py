def degrade_input(df):
    df = df.copy()
    df["Load_kN"] *= 1.25
    df["Length_m"] *= 0.9
    return df

def extreme_input(df):
    df = df.copy()
    df["Load_kN"] *= 1.6
    df["Length_m"] *= 0.7
    return df
