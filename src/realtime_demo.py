import os
import sys
import time

import numpy as np
import torch
import sounddevice as sd

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from config import SAMPLE_RATE
from src.model import MaskNet
from src.fxlms import FxLMS


# ============================================================
# Configuration
# ============================================================

N_FFT = 512
HOP = 160

BLOCK_MS = 20


# ============================================================
# CPU configuration
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

# Reduce CPU thread overhead during realtime processing
if device == "cpu":
    torch.set_num_threads(1)


# ============================================================
# Load neural model
# ============================================================

model = MaskNet(
    n_freq=N_FFT // 2 + 1
).to(device)

ckpt = "checkpoints/masknet.pt"

if os.path.exists(ckpt):

    model.load_state_dict(
        torch.load(
            ckpt,
            map_location=device
        )
    )

    print(
        "Loaded checkpoint:",
        ckpt
    )

else:

    print(
        "WARNING: checkpoint not found."
    )

model.eval()


# ============================================================
# FxLMS
# ============================================================

canceller = FxLMS(
    n_taps=64,
    mu=0.0001
)


# ============================================================
# STFT window
# ============================================================

hann = torch.hann_window(
    N_FFT
).to(device)


# ============================================================
# Neural processing
# ============================================================

def process_frame(
    noisy_frame: np.ndarray
) -> np.ndarray:

    original_length = len(
        noisy_frame
    )

    # --------------------------------------------------------
    # Convert NumPy -> Torch
    # --------------------------------------------------------

    x = torch.from_numpy(
        noisy_frame
    ).float().to(device)

    x = x.unsqueeze(0)

    # --------------------------------------------------------
    # Pad to FFT size
    # --------------------------------------------------------

    if original_length < N_FFT:

        pad_length = (
            N_FFT
            - original_length
        )

        x = torch.nn.functional.pad(
            x,
            (0, pad_length)
        )

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    X = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP,
        window=hann,
        return_complex=True,
        center=True
    )

    # --------------------------------------------------------
    # Neural mask
    # --------------------------------------------------------

    with torch.inference_mode():

        magnitude = torch.abs(X)

        mask = model(
            magnitude
        )

    # --------------------------------------------------------
    # Apply mask
    # --------------------------------------------------------

    enhanced_spec = (
        X * mask
    )

    # --------------------------------------------------------
    # ISTFT
    # --------------------------------------------------------

    enhanced = torch.istft(
        enhanced_spec,
        n_fft=N_FFT,
        hop_length=HOP,
        window=hann,
        length=x.shape[-1],
        center=True
    )

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    enhanced = (
        enhanced
        .squeeze(0)
        [:original_length]
    )

    enhanced_np = (
        enhanced
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return enhanced_np


# ============================================================
# Realtime
# ============================================================

def run(
    block_ms=20,
    mode="hybrid"
):

    block_size = int(
        SAMPLE_RATE
        * block_ms
        / 1000
    )

    print()
    print(
        "Starting real-time loop"
    )

    print(
        f"Sample rate : {SAMPLE_RATE}"
    )

    print(
        f"Block size  : {block_size}"
    )

    print(
        f"Block time  : {block_ms} ms"
    )

    print(
        f"Mode        : {mode}"
    )

    print()

    # --------------------------------------------------------
    # Callback
    # --------------------------------------------------------

    def callback(
        indata,
        outdata,
        frames,
        time_info,
        status
    ):

        # Do not print status continuously
        # inside realtime callback.

        noisy = (
            indata[:, 0]
            .copy()
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # OFF
        # ----------------------------------------------------

        if mode == "off":

            out = noisy

        # ----------------------------------------------------
        # CLASSICAL
        # ----------------------------------------------------

        elif mode == "classical_only":

            # IMPORTANT:
            #
            # A single microphone does not provide a real
            # independent reference signal.
            #
            # Therefore we do NOT adapt FxLMS against the
            # complete speech signal.
            #
            # Keep adaptation disabled here.

            _, e = canceller.process_block(
                noisy,
                noisy
            )

            out = noisy

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        else:

            # Neural enhancement
            enhanced = process_frame(
                noisy
            )

            # Estimate the component removed
            # by the neural network.
            estimated_noise = (
                noisy
                - enhanced
            )

            # Run FxLMS on the estimated noise
            # for adaptive filtering / monitoring.
            #
            # We DO NOT inject the FxLMS result
            # directly into the speaker output because
            # there is currently no independent reference
            # microphone.

            _, e = canceller.process_block(
                estimated_noise,
                estimated_noise
            )

            # Neural enhanced audio is the actual output.
            out = enhanced

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        out = np.asarray(
            out,
            dtype=np.float32
        )

        # Prevent accidental clipping
        out = np.clip(
            out,
            -1.0,
            1.0
        )

        outdata[:, 0] = out

    # ========================================================
    # Audio stream
    # ========================================================

    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=block_size,
        channels=1,
        dtype="float32",
        callback=callback
    ):

        print(
            "Running..."
        )

        print(
            "Press Ctrl+C to stop."
        )

        try:

            while True:

                time.sleep(
                    0.5
                )

        except KeyboardInterrupt:

            print()
            print(
                "Stopped."
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "off",
            "classical_only",
            "hybrid"
        ],
        default="hybrid"
    )

    parser.add_argument(
        "--block_ms",
        type=int,
        default=20
    )

    args = parser.parse_args()

    run(
        block_ms=args.block_ms,
        mode=args.mode
    )