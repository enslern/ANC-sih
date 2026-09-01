import os
import sys
import time
import numpy as np
import torch
import sounddevice as sd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SAMPLE_RATE, FRAME_LEN
from src.model import MaskNet
from src.fxlms import FxLMS

N_FFT = 512
HOP = 160

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaskNet(n_freq=N_FFT // 2 + 1).to(device)
ckpt = "checkpoints/masknet.pt"
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print("Loaded checkpoint:", ckpt)
else:
    print("WARNING: no checkpoint found, using untrained weights (demo only).")
model.eval()

canceller = FxLMS(n_taps=128, mu=0.01)

hann = torch.hann_window(N_FFT).to(device)


def process_frame(noisy_frame: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(noisy_frame).float().to(device).unsqueeze(0)
    X = torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=hann,
                    return_complex=True, center=False)
    with torch.no_grad():
        mask = model(torch.abs(X))
    Enh = X * mask
    enhanced = torch.istft(Enh, n_fft=N_FFT, hop_length=HOP, window=hann,
                            length=len(noisy_frame), center=False)
    residual = noisy_frame - enhanced.squeeze(0).cpu().numpy()

    # feed residual into FxLMS as the "reference" for the leftover stationary noise
    _, e = canceller.process_block(residual, residual)
    out = enhanced.squeeze(0).cpu().numpy() + e * 0.0  # e is error signal, kept for logging
    return out.astype(np.float32)


def run(block_ms=20, mode="hybrid"):
    block_size = int(SAMPLE_RATE * block_ms / 1000)
    print(f"Starting real-time loop, block={block_size} samples ({block_ms}ms), mode={mode}")

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status)
        noisy = indata[:, 0].astype(np.float32)
        if mode == "off":
            out = noisy
        elif mode == "classical_only":
            _, e = canceller.process_block(noisy, noisy)
            out = e.astype(np.float32)
        else:  # hybrid
            out = process_frame(noisy)
        outdata[:, 0] = out

    with sd.Stream(samplerate=SAMPLE_RATE, blocksize=block_size,
                    channels=1, dtype="float32", callback=callback):
        print("Running... Ctrl+C to stop.")
        while True:
            time.sleep(0.5)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["off", "classical_only", "hybrid"], default="hybrid")
    p.add_argument("--block_ms", type=int, default=20)
    args = p.parse_args()
    run(block_ms=args.block_ms, mode=args.mode)