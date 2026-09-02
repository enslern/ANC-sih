# extract_esc50_noise.py
import os
import shutil
import pandas as pd

ESC_AUDIO = "data/ESC-50-master/audio"
ESC_META = "data/ESC-50-master/meta/esc50.csv"
DST = "data/noise"

os.makedirs(DST, exist_ok=True)
TARGET_CATEGORIES = ["helicopter", "engine", "siren", "airplane", "car_horn", "train"]

df = pd.read_csv(ESC_META)
matches = df[df["category"].isin(TARGET_CATEGORIES)]
print(f"Found {len(matches)} matching clips across categories: {matches['category'].unique()}")

for _, row in matches.iterrows():
    src = os.path.join(ESC_AUDIO, row["filename"])
    dst = os.path.join(DST, f"{row['category']}_{row['filename']}")
    shutil.copy(src, dst)

print("Copied to", DST)