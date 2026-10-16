
import pandas as pd
from pathlib import Path

# Create the data folder
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Dataset file on Hugging Face
dataset_url = (
    "hf://datasets/rajpurkar/squad/"
    "plain_text/train-00000-of-00001.parquet"
)

print("Downloading and loading the SQuAD training dataset...")

try:
    df = pd.read_parquet(dataset_url)

    print("Dataset loaded successfully!")
    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())

    # Save the dataset locally
    output_file = data_dir / "squad_train.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Dataset saved to: {output_file.resolve()}")
    print("\nFirst five rows:")
    print(df.head())

except Exception as e:
    print("Dataset download failed.")
    print("Error:", e)
    print("Check your internet connection and installed libraries.")
