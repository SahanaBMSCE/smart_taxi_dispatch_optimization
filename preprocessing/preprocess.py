import pandas as pd

df = pd.read_csv("data/raw/taxi_data.csv")

df.to_csv(
    "data/processed/clean.csv",
    index=False
)

print("Preprocessing complete")