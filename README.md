# Hybrid AI/ML + FxLMS ANC — Hackathon Prototype

An adaptive noise cancellation pipeline combining a lightweight neural suppression model with a classical FxLMS adaptive filter, targeting defence communication use cases (gunfire, rotor, engine, siren noise).

## Table of Contents

- [Folder Structure](#folder-structure)
- [1. Install Dependencies](#1-install-dependencies)
- [2. Build the Dataset Locally](#2-build-the-dataset-locally-not-tracked-in-git)
- [3. Train the Suppression Model](#3-train-the-suppression-model)
- [4. Sanity-Check the Classical FxLMS Filter](#4-sanity-check-the-classical-fxlms-filter-alone)
- [5. Run the Live Demo](#5-run-the-live-demo-needs-a-mic--speakerheadset)
- [6. Evaluate Against the Brief's Targets](#6-evaluate-against-the-briefs-targets)
- [7. Embedded Deployment (Jetson Orin Nano)](#7-embedded-deployment-jetson-orin-nano)
- [Troubleshooting Notes from Setup](#troubleshooting-notes-from-setup)
- [8. Local API + UI Demo](#8-local-api--ui-demo)
  - [8.1 Test the API](#81-test-the-api)
  - [8.2 Run the Model on a Local API Server](#82-run-the-model-on-a-local-api-server)
  - [8.3 Run the UI Locally](#83-run-the-ui-locally)
  - [8.4 Complete Local Demo Workflow](#84-complete-local-demo-workflow)
  - [8.5 Troubleshooting the Local API + UI](#85-troubleshooting-the-local-api--ui)

## Folder Structure

```text
ANC-sih/
├── requirements.txt
├── README.md
├── config.py
├── convert_librispeech.py
├── extract_esc50_noise.py
├── .gitignore
├── api.py
├── test_api.py
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
```

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note (Windows):** `pesq` is commented out in `requirements.txt` because it needs Microsoft C++ Build Tools to compile on Windows. The brief's SNR and STOI targets don't need it — only PESQ does. Install [build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) later if you want PESQ scores.

> **Note:** if you hit `ModuleNotFoundError: No module named 'pkg_resources'` when importing librosa, run `pip install "setuptools<81"` — newer setuptools versions dropped `pkg_resources`, which the installed librosa version still needs.

## 2. Build the Dataset Locally (not tracked in git)

Data files are excluded from git via `.gitignore` — every teammate builds their own copy locally, once, using the same public sources so everyone ends up with identical data.

### a) Clean Speech — LibriSpeech dev-clean

Download: [`dev-clean.tar.gz`](https://www.openslr.org/resources/12/dev-clean.tar.gz)

```bash
tar -xzf dev-clean.tar.gz -C data/
```

This creates `data/LibriSpeech/dev-clean/<speaker>/<chapter>/*.flac`.

Convert to 16kHz WAV:

```bash
python convert_librispeech.py
```

Verify:

```powershell
Get-ChildItem "data\clean" -Filter "*.wav" | Measure-Object
# expect ~2703
```

### b) Noise Clips — ESC-50 (helicopter/engine/siren/airplane/car_horn/train)

Download: [ESC-50](https://github.com/karoldvl/ESC-50/archive/master.zip)

Save as `esc50.zip` in the project root, then:

```powershell
Expand-Archive -Path "esc50.zip" -DestinationPath "data\"
python extract_esc50_noise.py
```

Verify:

```powershell
Get-ChildItem "data\noise" -Filter "*.wav" | Measure-Object
# expect ~200-240
```

> **Still missing:** gunshot/artillery clips. ESC-50 doesn't cover these well. Source 10-20 CC0/CC-BY clips from [Freesound.org](https://freesound.org) (search "gunshot", "artillery", "explosion") and drop the `.wav` files directly into `data/noise/`. Not blocking for the rest of the pipeline — noise-type variety just improves the final model/report.

### c) Generate Noisy/Clean Training Pairs

```bash
python -m src.dataset_gen
```

Writes paired examples to `data/generated/noisy/` and `data/generated/clean/`. Takes a few minutes for the default 2000 examples.

## 3. Train the Suppression Model

```bash
python -m src.train
```

Saves the best checkpoint (by validation STOI) to `checkpoints/masknet.pt`.

## 4. Sanity-Check the Classical FxLMS Filter Alone

```bash
python -m src.fxlms
```

Should print error power dropping after convergence on a synthetic tone.

## 5. Run the Live Demo (needs a mic + speaker/headset)

```bash
python -m src.realtime_demo --mode hybrid --block_ms 20
```

**Modes:**

| Mode | Description |
|---|---|
| `off` | Passthrough (no processing) |
| `classical_only` | FxLMS alone |
| `hybrid` | Model + FxLMS |

Use these three modes for a live "noise on / ANC off / ANC on" A-B demo for judges.

## 6. Evaluate Against the Brief's Targets

Use `src/metrics.py`'s `evaluate(clean, enhanced, sr)` on a held-out test set:

| Metric | Target | Needs pesq? |
|---|---|---|
| SNR improvement | > 15 dB | No |
| STOI (intelligibility) | > 0.85 | No |
| PESQ (perceptual quality) | > 2.5 | Yes |

## 7. Embedded Deployment (Jetson Orin Nano)

Export the trained model to ONNX:

```python
import torch
from src.model import MaskNet

m = MaskNet()
m.load_state_dict(torch.load("checkpoints/masknet.pt"))
dummy = torch.randn(1, 257, 100)
torch.onnx.export(
    m, dummy, "masknet.onnx", opset_version=17,
    input_names=["mag"], output_names=["mask"],
    dynamic_axes={"mag": {2: "time"}, "mask": {2: "time"}}
)
```

Then convert `masknet.onnx` → TensorRT engine on the Jetson using `trtexec` or `torch2trt`, per Section 4.3 of the build guide.

## Troubleshooting Notes from Setup

- If `pip install pesq` fails with a Visual C++ error, skip it (see above) — it's optional for now.
- Windows PowerShell's `Invoke-WebRequest` can hang on large GitHub zip downloads; prefer `curl.exe -L -o file.zip "<url>"` or download manually via browser instead.
- Data files (`data/`, `*.wav`, `*.flac`, `checkpoints/`) are gitignored — do not commit them. If you accidentally do, `git rm -r --cached data` removes them from tracking without deleting your local copies.

---

## 8. Local API + UI Demo

The trained MaskNet model can also be accessed through a local FastAPI server and tested through the frontend UI.

The local demo uses:

```text
test_api.py
      ↓
   api.py
      ↓
MaskNet model
      ↓
  React UI
```

The API and UI are intended for the hackathon prototype. The UI allows audio to be uploaded or recorded, processed by the actual trained MaskNet model, and the enhanced audio to be played back.

### 8.1 Test the API

Before using the UI, test the API setup with:

```bash
python test_api.py
```

This verifies that the API endpoints are working correctly.

The API exposes:

```http
GET  /api/anc/status
POST /api/anc/process
```

If the API server is not running, start it first using [Section 8.2](#82-run-the-model-on-a-local-api-server).

### 8.2 Run the Model on a Local API Server

The FastAPI server is implemented in `api.py`. The server loads the trained model checkpoint from `checkpoints/masknet.pt`.

Start the API server from the `ANC-sih` directory:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The server will run at `http://localhost:8000`.

Check the model status by opening:

```text
http://localhost:8000/api/anc/status
```

The audio processing endpoint is:

```text
http://localhost:8000/api/anc/process
```

Keep this terminal running while using the UI. The model is loaded once when the server starts and is reused for subsequent audio-processing requests.

### 8.3 Run the UI Locally

The frontend UI is located in the separate project:

```text
C:\Users\Elridge fernandes\Downloads\anc-audio-enhancer
```

Open a **second terminal**, go to the UI directory, and run:

```powershell
cd "C:\Users\Elridge fernandes\Downloads\anc-audio-enhancer"
npm install   # first time only
npm run dev
```

The UI will normally be available at `http://localhost:3000`. Open the address in a browser.

### 8.4 Complete Local Demo Workflow

Use two terminals.

**Terminal 1 — Start the MaskNet API**

```powershell
cd "C:\path\to\ANC-sih"
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Verify: `http://localhost:8000/api/anc/status`

**Terminal 2 — Start the UI**

```powershell
cd "C:\Users\Elridge fernandes\Downloads\anc-audio-enhancer"
npm run dev
```

Then open `http://localhost:3000`.

**Demo flow:**

1. Start the Python API
2. Verify the API status
3. Start the React UI
4. Open `localhost:3000`
5. Upload or record an audio sample
6. Send the audio for enhancement
7. MaskNet processes the audio
8. Listen to the enhanced output
9. View the processing metrics

The UI communicates with the Python backend at `http://localhost:8000/api/anc/process`. The enhancement is performed by the trained MaskNet model rather than a frontend mock model.

### 8.5 Troubleshooting the Local API + UI

<details>
<summary><strong>npm is not recognized</strong></summary>

Install Node.js and restart VS Code.

Check:

```bash
node --version
npm --version
```

Then run:

```bash
npm install
npm run dev
```
</details>

<details>
<summary><strong>UI cannot connect to the backend</strong></summary>

Make sure the FastAPI server is running in the first terminal:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Then verify: `http://localhost:8000/api/anc/status`
</details>

<details>
<summary><strong>Port 8000 is already in use</strong></summary>

Start the API on another port:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

If you change the API port, update the frontend API endpoint accordingly.
</details>

<details>
<summary><strong>Port 3000 is already in use</strong></summary>

The Vite frontend may automatically select another available port. Open the URL shown in the terminal.
</details>

<details>
<summary><strong>Model checkpoint not found</strong></summary>

Make sure the trained checkpoint exists at `checkpoints/masknet.pt`. The API must be started from the root `ANC-sih` directory so that the relative checkpoint path resolves correctly.
</details>