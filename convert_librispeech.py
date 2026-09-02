# convert_librispeech.py
import os
import glob
import soundfile as sf
import librosa

SRC = "data/LibriSpeech/dev-clean"
DST = "data/clean"
TARGET_SR = 16000

os.makedirs(DST, exist_ok=True)
flac_files = glob.glob(os.path.join(SRC, "**", "*.flac"), recursive=True)
print(f"Found {len(flac_files)} flac files")

for i, f in enumerate(flac_files):
    audio, sr = sf.read(f, dtype="float32")
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    out_name = os.path.basename(f).replace(".flac", ".wav")
    sf.write(os.path.join(DST, out_name), audio, TARGET_SR)
    if i % 500 == 0:
        print(f"[{i}/{len(flac_files)}] converted")

print("Done.")