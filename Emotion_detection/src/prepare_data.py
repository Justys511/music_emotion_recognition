import pandas as pd
import os

# Paths
INPUT_CSV = "data/Emotify/dataset.csv"  # Source file
DATA_DIR = "data/Emotify"  # Folder with audio files
OUTPUT_CSV = "data/processed_dataset.csv"  # Processing result

# Load the dataset
df = pd.read_csv(INPUT_CSV)

# Check for the required columns
required_columns = ["track_id", "genre"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Error: The dataset is missing the column '{col}'.")

# Remove duplicates to enable correct mapping
unique_tracks = df[["track_id", "genre"]].drop_duplicates()

# Create local numbering for each genre
unique_tracks["local_id"] = unique_tracks.groupby("genre").cumcount() + 1

# Mapping from track_id -> local_id
track_to_local_id = unique_tracks.set_index(["track_id", "genre"])["local_id"].to_dict()

# Add local numbering to the main dataset
df["local_id"] = df.apply(lambda row: track_to_local_id.get((row["track_id"], row["genre"]), None), axis=1)

# Check for any problems with missing local_id
if df["local_id"].isna().any():
    raise ValueError("Error: Failed to generate local_id for some records.")

# Form the path to files
df["file_name"] = df["local_id"].astype(int).astype(str) + ".wav"
df["file_path"] = df.apply(lambda row: os.path.join(DATA_DIR, row["genre"], row["file_name"]), axis=1)

# Check for file existence
df["file_exists"] = df["file_path"].apply(os.path.exists)

# Display missing files
missing_files = df[~df["file_exists"]]
if not missing_files.empty:
    print("⚠️ Missing files:")
    print(missing_files[["track_id", "genre", "file_path"]].head(20))

# Remove records with missing files
df = df[df["file_exists"]]

# Remove temporary columns before saving
df = df.drop(columns=["file_exists", "local_id"])

# Save the processed dataset
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)


# For example, your dataset after preparation:
df = pd.read_csv("data/processed_dataset.csv")

emotion_columns = [
    "amazement", "solemnity", "tenderness", "nostalgia", "calmness",
    "power", "joyful_activation", "tension", "sadness"
]

# Sum over the columns (how many times "1" appears)
counts = df[emotion_columns].sum().sort_values(ascending=False)

print("Number of examples for each emotion (how many '1' entries there are):")
print(counts)
