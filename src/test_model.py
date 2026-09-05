import os
import sys
import numpy as np
import torch
import soundfile as sf

from src.model import MaskNet


# ============================================================
# CONFIGURATION
# ============================================================

N_FFT = 512
HOP = 160
SAMPLE_RATE = 16000

CHECKPOINT = "checkpoints/masknet.pt"

# Put your noisy audio here
INPUT_AUDIO = "data/my_noisy_audio.wav"

# Output file
OUTPUT_AUDIO = "data/my_enhanced_audio.wav"


# ============================================================
# LOAD MODEL
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = MaskNet(
    n_freq=N_FFT // 2 + 1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print("Loaded model on:", DEVICE)


# ============================================================
# LOAD AUDIO
# ============================================================

if not os.path.exists(INPUT_AUDIO):
    print(f"\nERROR: Audio file not found:")
    print(INPUT_AUDIO)
    sys.exit(1)

noisy, sr = sf.read(
    INPUT_AUDIO,
    dtype="float32"
)

print("\nInput audio:")
print("  File:", INPUT_AUDIO)
print("  Sample rate:", sr)
print("  Duration:", len(noisy) / sr, "seconds")


# ============================================================
# HANDLE STEREO
# ============================================================

if noisy.ndim > 1:
    print("  Stereo detected → converting to mono")
    noisy = np.mean(noisy, axis=1)


# ============================================================
# RESAMPLE CHECK
# ============================================================

if sr != SAMPLE_RATE:

    print(
        f"\nERROR: Model expects {SAMPLE_RATE} Hz audio."
    )

    print(
        f"Your file is {sr} Hz."
    )

    print(
        "\nConvert the audio to 16 kHz WAV before testing."
    )

    sys.exit(1)


# ============================================================
# RUN MASKNET
# ============================================================

print("\nRunning MaskNet...")

x = torch.from_numpy(noisy).float()

x = x.to(DEVICE).unsqueeze(0)

window = torch.hann_window(
    N_FFT,
    device=DEVICE
)

with torch.no_grad():

    # STFT
    X = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP,
        window=window,
        return_complex=True
    )

    # Magnitude
    magnitude = torch.abs(X)

    # Neural noise-suppression mask
    mask = model(magnitude)

    # Apply mask
    enhanced_spec = X * mask

    # Back to waveform
    enhanced = torch.istft(
        enhanced_spec,
        n_fft=N_FFT,
        hop_length=HOP,
        window=window,
        length=len(noisy)
    )


# ============================================================
# SAVE RESULT
# ============================================================

enhanced = enhanced.squeeze(0).cpu().numpy()

sf.write(
    OUTPUT_AUDIO,
    enhanced,
    SAMPLE_RATE
)


# ============================================================
# BASIC SIGNAL STATISTICS
# ============================================================

input_rms = np.sqrt(
    np.mean(noisy ** 2)
)

output_rms = np.sqrt(
    np.mean(enhanced ** 2)
)

input_peak = np.max(
    np.abs(noisy)
)

output_peak = np.max(
    np.abs(enhanced)
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("                 MASKNET TEST")
print("=" * 60)

print("\nInput:")
print(" ", INPUT_AUDIO)

print("\nOutput:")
print(" ", OUTPUT_AUDIO)

print("\nSignal levels:")
print(f"  Input RMS:  {input_rms:.6f}")
print(f"  Output RMS: {output_rms:.6f}")

print(f"\n  Input peak:  {input_peak:.6f}")
print(f"  Output peak: {output_peak:.6f}")

print("\nDone!")
print("=" * 60)