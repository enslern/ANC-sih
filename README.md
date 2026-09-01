# Hybrid AI/ML + FxLMS ANC — Hackathon Prototype

## Setup

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## 1. Populate data
Put clean speech wavs (LibriSpeech/VCTK subset or your own recordings) in `data/clean/`,
and defence-noise wavs (gunshot, rotor, engine, siren, artillery) in `data/noise/`.

## 2. Generate the training dataset

    python -m src.dataset_gen

Writes noisy/clean pairs to `data/generated/`.

## 3. Train the suppression model

    python -m src.train

Saves the best checkpoint (by validation STOI) to `checkpoints/masknet.pt`.

## 4. Sanity-check the classical FxLMS filter alone

    python -m src.fxlms

Should show error power dropping after convergence on a synthetic tone.

## 5. Run the live demo (needs a mic + speaker/headset)

    python -m src.realtime_demo --mode hybrid --block_ms 20

Modes: `off` (passthrough), `classical_only` (FxLMS alone), `hybrid` (model + FxLMS) —
use these to do the "noise on / ANC off / ANC on" A-B demo for judges.

## 6. Evaluate against the brief's targets
Use `src/metrics.py`'s `evaluate(clean, enhanced, sr)` on your held-out test set to report
SNR (>15 dB), STOI (>0.85), PESQ (>2.5).

## Embedded deployment (Jetson)
Export the trained model to ONNX, then convert with `trtexec`/`torch2trt` to a TensorRT
engine, per Section 4.3 of the build guide. The `MaskNet` in `src/model.py` is plain
Conv1d/GRU/Linear, so it exports cleanly:

    import torch
    from src.model import MaskNet
    m = MaskNet(); m.load_state_dict(torch.load("checkpoints/masknet.pt"))
    dummy = torch.randn(1, 257, 100)
    torch.onnx.export(m, dummy, "masknet.onnx", opset_version=17,
                       input_names=["mag"], output_names=["mask"],
                       dynamic_axes={"mag": {2: "time"}, "mask": {2: "time"}})