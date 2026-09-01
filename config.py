import os

SAMPLE_RATE = 16000
FRAME_MS = 20  # 20ms frames -> 320 samples at 16kHz, matches real-time targets
FRAME_LEN = int(SAMPLE_RATE * FRAME_MS / 1000)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
GEN_DIR = os.path.join(DATA_DIR, "generated")

SNR_RANGE_DB = (-5, 15)

# FxLMS
FXLMS_TAPS = 128
FXLMS_MU = 0.01

# Targets from the brief
TARGET_SNR_DB = 15
TARGET_STOI = 0.85
TARGET_PESQ = 2.5