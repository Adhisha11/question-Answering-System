
import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parent / "data" / "squad_train.csv"

print("CSV exists:", file_path.exists())

if file_path.exists():
    df = pd.read_csv(file_path)

    print("\nDataset shape:", df.shape)
    print("\nColumn names:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    for column in ["question", "context", "answers"]:
        if column in df.columns:
            print(f"\n{column}:")
            print("Non-empty rows:", df[column].notna().sum())
            print("Example:", repr(df[column].dropna().head(1).tolist()))
        else:
            print(f"\nMISSING COLUMN: {column}")
else:
    print("ERROR: squad_train.csv was not found.")
    print("Run download_dataset.py first.")
