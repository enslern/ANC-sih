# Hybrid AI/ML + FxLMS ANC — Hackathon Prototype

An adaptive noise cancellation pipeline combining a lightweight neural
suppression model with a classical FxLMS adaptive filter, targeting
defence communication use cases (gunfire, rotor, engine, siren noise).

## Folder structure

    ANC-sih/
    ├── requirements.txt
    ├── README.md
    ├── config.py
    ├── convert_librispeech.py
    ├── extract_esc50_noise.py
    ├── .gitignore
    ├── src/
    │   ├── __init__.py
    │   ├── dataset_gen.py
    │   ├── model.py
    │   ├── train.py
    │   ├── metrics.py
    │   ├── fxlms.py
    │   └── realtime_demo.py
    └── data/                  <- NOT in git, built locally (see below)
        ├── clean/
        ├── noise/
        └── generated/

## 1. Install dependencies

    pip install -r requirements.txt

> **Note (Windows):** `pesq` is commented out in requirements.txt because it
> needs Microsoft C++ Build Tools to compile on Windows. The brief's SNR and
> STOI targets don't need it — only PESQ does. Install build tools later if
> you want PESQ scores: https://visualstudio.microsoft.com/visual-cpp-build-tools/

> **Note:** if you hit `ModuleNotFoundError: No module named 'pkg_resources'`
> when importing librosa, run `pip install "setuptools<81"` — newer setuptools
> versions dropped `pkg_resources`, which the installed librosa version still needs.

## 2. Build the dataset locally (not tracked in git)

Data files are excluded from git via `.gitignore` — every teammate builds
their own copy locally, once, using the same public sources so everyone
ends up with identical data.

### a) Clean speech — LibriSpeech dev-clean

Download: https://www.openslr.org/resources/12/dev-clean.tar.gz

    tar -xzf dev-clean.tar.gz -C data/

This creates `data/LibriSpeech/dev-clean/<speaker>/<chapter>/*.flac`.

Convert to 16kHz WAV:

    python convert_librispeech.py

Verify:

    Get-ChildItem "data\clean" -Filter "*.wav" | Measure-Object
    # expect ~2703

### b) Noise clips — ESC-50 (helicopter/engine/siren/airplane/car_horn/train)

Download: https://github.com/karoldvl/ESC-50/archive/master.zip
Save as `esc50.zip` in the project root, then:

    Expand-Archive -Path "esc50.zip" -DestinationPath "data\"
    python extract_esc50_noise.py

Verify:

    Get-ChildItem "data\noise" -Filter "*.wav" | Measure-Object
    # expect ~200-240

> **Still missing:** gunshot/artillery clips. ESC-50 doesn't cover these
> well. Source 10-20 CC0/CC-BY clips from Freesound.org (search "gunshot",
> "artillery", "explosion") and drop the `.wav` files directly into
> `data/noise/`. Not blocking for the rest of the pipeline — noise-type
> variety just improves the final model/report.

### c) Generate noisy/clean training pairs

    python -m src.dataset_gen

Writes paired examples to `data/generated/noisy/` and `data/generated/clean/`.
Takes a few minutes for the default 2000 examples.

## 3. Train the suppression model

    python -m src.train

Saves the best checkpoint (by validation STOI) to `checkpoints/masknet.pt`.

## 4. Sanity-check the classical FxLMS filter alone

    python -m src.fxlms

Should print error power dropping after convergence on a synthetic tone.

## 5. Run the live demo (needs a mic + speaker/headset)

    python -m src.realtime_demo --mode hybrid --block_ms 20

Modes:
- `off` — passthrough (no processing)
- `classical_only` — FxLMS alone
- `hybrid` — model + FxLMS

Use these three modes for a live "noise on / ANC off / ANC on" A-B demo for judges.

## 6. Evaluate against the brief's targets

Use `src/metrics.py`'s `evaluate(clean, enhanced, sr)` on a held-out test set:

| Metric | Target | Needs pesq? |
|---|---|---|
| SNR improvement | > 15 dB | No |
| STOI (intelligibility) | > 0.85 | No |
| PESQ (perceptual quality) | > 2.5 | Yes |

## 7. Embedded deployment (Jetson Orin Nano)

Export the trained model to ONNX:

    import torch
    from src.model import MaskNet
    m = MaskNet(); m.load_state_dict(torch.load("checkpoints/masknet.pt"))
    dummy = torch.randn(1, 257, 100)
    torch.onnx.export(m, dummy, "masknet.onnx", opset_version=17,
                       input_names=["mag"], output_names=["mask"],
                       dynamic_axes={"mag": {2: "time"}, "mask": {2: "time"}})

Then convert `masknet.onnx` → TensorRT engine on the Jetson using
`trtexec` or `torch2trt`, per Section 4.3 of the build guide.

## Troubleshooting notes from setup

- If `pip install pesq` fails with a Visual C++ error, skip it (see above) —
  it's optional for now.
- Windows PowerShell's `Invoke-WebRequest` can hang on large GitHub zip
  downloads; prefer `curl.exe -L -o file.zip "<url>"` or download manually
  via browser instead.
- Data files (`data/`, `*.wav`, `*.flac`, `checkpoints/`) are gitignored —
  do not commit them. If you accidentally do, `git rm -r --cached data`
  removes them from tracking without deleting your local copies.