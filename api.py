import base64
import io
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import torch

from pydantic import BaseModel
from src.model import MaskNet

# ============================================================
# CONFIGURATION
# ============================================================

N_FFT = 512
HOP = 160
SAMPLE_RATE = 16000
CHECKPOINT = "checkpoints/masknet.pt"

# ============================================================
# FASTAPI & CORS (INITIALIZED ONCE ONLY)
# ============================================================

app = FastAPI(title="ANC MaskNet Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Vercel, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],  # Allows Content-Type and custom headers
)


# Catch-all OPTIONS preflight handler to prevent 400 Bad Request
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
  return JSONResponse(
      content="OK",
      headers={
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
          "Access-Control-Allow-Headers": "*",
      },
  )


# ============================================================
# DEVICE & MODEL LOADING
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "=" * 60)
print("Loading MaskNet")
print("=" * 60)
print("Device:", DEVICE)

model = MaskNet(n_freq=N_FFT // 2 + 1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

print("Checkpoint:", CHECKPOINT)
print("MaskNet loaded successfully.")
print("=" * 60 + "\n")


# ============================================================
# REQUEST SCHEMAS
# ============================================================


class AudioMetadata(BaseModel):
  filename: str
  duration: float
  sampleRate: int
  channels: int


class AncProcessRequest(BaseModel):
  audioBase64: str
  audioMetadata: AudioMetadata
  referenceAudioBase64: str | None = None


# ============================================================
# ROUTES
# ============================================================


@app.get("/api/anc/status")
def health_check():
  return {
      "status": "online",
      "backend": "Python FastAPI + PyTorch",
      "model": "MaskNet",
      "device": DEVICE,
      "sampleRate": SAMPLE_RATE,
      "fftSize": N_FFT,
      "hopSize": HOP,
      "checkpoint": CHECKPOINT,
  }


@app.post("/api/anc/process")
async def process_audio(req: AncProcessRequest):
  total_start = time.perf_counter()

  try:
    # 1. Decode Base64
    try:
      audio_bytes = base64.b64decode(req.audioBase64)
    except Exception as e:
      return {"success": False, "error": f"Invalid base64 audio: {e}"}

    # 2. Read WAV
    noisy, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # 3. Stereo -> Mono
    if noisy.ndim > 1:
      noisy = np.mean(noisy, axis=1)

    # 4. Verify Sample Rate
    if sr != SAMPLE_RATE:
      return {
          "success": False,
          "error": (
              f"Expected {SAMPLE_RATE} Hz audio, received {sr} Hz."
          ),
      }

    noisy = np.asarray(noisy, dtype=np.float32)
    num_samples = len(noisy)
    duration = num_samples / SAMPLE_RATE

    if num_samples == 0:
      return {"success": False, "error": "Audio file is empty."}

    input_rms = float(np.sqrt(np.mean(noisy**2)))
    input_peak = float(np.max(np.abs(noisy)))

    # 5. PyTorch Tensors
    x = torch.from_numpy(noisy).float().to(DEVICE).unsqueeze(0)
    window = torch.hann_window(N_FFT, device=DEVICE)

    # 6. MaskNet Inference
    inference_start = time.perf_counter()

    with torch.no_grad():
      X = torch.stft(
          x,
          n_fft=N_FFT,
          hop_length=HOP,
          window=window,
          return_complex=True,
      )
      magnitude = torch.abs(X)
      mask = model(magnitude)
      enhanced_spec = X * mask
      enhanced = torch.istft(
          enhanced_spec,
          n_fft=N_FFT,
          hop_length=HOP,
          window=window,
          length=num_samples,
      )

    inference_time = time.perf_counter() - inference_start

    # 7. Format Output
    enhanced = enhanced.squeeze(0).cpu().numpy().astype(np.float32)

    output_rms = float(np.sqrt(np.mean(enhanced**2)))
    output_peak = float(np.max(np.abs(enhanced)))

    output_buffer = io.BytesIO()
    sf.write(
        output_buffer,
        enhanced,
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )

    enhanced_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

    total_time = time.perf_counter() - total_start
    processing_time_ms = total_time * 1000
    inference_time_ms = inference_time * 1000
    real_time_factor = (
        inference_time / duration if duration > 0 else 0
    )

    return {
        "success": True,
        "enhancedAudioBase64": enhanced_base64,
        "enhancedAudioFormat": "audio/wav",
        "metrics": {
            "inputSnr": None,
            "outputSnr": None,
            "snrImprovement": None,
            "siSdr": None,
            "stoi": None,
            "pesq": None,
            "processingTimeMs": round(processing_time_ms, 2),
            "inferenceTimeMs": round(inference_time_ms, 2),
            "realTimeFactor": round(real_time_factor, 3),
            "hasReference": False,
            "inputRms": input_rms,
            "outputRms": output_rms,
            "metricNotes": {
                "notice": (
                    "Reference-based metrics require aligned clean audio."
                )
            },
        },
        "modelInfo": {
            "modelName": "MaskNet",
            "version": "1.0",
            "architecture": "Neural Spectral Masking",
            "sampleRate": SAMPLE_RATE,
            "fftSize": N_FFT,
            "hopSize": HOP,
            "device": DEVICE,
        },
    }

  except Exception as e:
    print("\nMASKNET INFERENCE ERROR:")
    print(e)
    print()
    return {
        "success": False,
        "error": f"MaskNet inference failed: {str(e)}",
    }