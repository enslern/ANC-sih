import io
import time
import base64

import numpy as np
import torch
import soundfile as sf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
# FASTAPI
# ============================================================

app = FastAPI(
    title="ANC MaskNet Backend",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for Vercel frontend and preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Vercel, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],  # Allows Content-Type and custom headers
)

# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 60)
print("Loading MaskNet")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

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

print("Checkpoint:", CHECKPOINT)
print("MaskNet loaded successfully.")
print("=" * 60)
print()


# ============================================================
# REQUEST SCHEMA
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
# HEALTH CHECK
# ============================================================

@app.get("/api/anc/status")
def health_check():

    return {

        "status": "online",

        "backend":
            "Python FastAPI + PyTorch",

        "model":
            "MaskNet",

        "device":
            DEVICE,

        "sampleRate":
            SAMPLE_RATE,

        "fftSize":
            N_FFT,

        "hopSize":
            HOP,

        "checkpoint":
            CHECKPOINT,

    }


# ============================================================
# PROCESS AUDIO
# ============================================================

@app.post("/api/anc/process")
async def process_audio(
    req: AncProcessRequest
):

    total_start = time.perf_counter()

    try:

        # ====================================================
        # DECODE BASE64 WAV
        # ====================================================

        try:

            audio_bytes = base64.b64decode(
                req.audioBase64
            )

        except Exception as e:

            return {
                "success": False,
                "error":
                    f"Invalid base64 audio: {e}"
            }


        # ====================================================
        # READ WAV
        # ====================================================

        noisy, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32"
        )


        # ====================================================
        # STEREO → MONO
        # ====================================================

        if noisy.ndim > 1:

            noisy = np.mean(
                noisy,
                axis=1
            )


        # ====================================================
        # VERIFY SAMPLE RATE
        # ====================================================

        if sr != SAMPLE_RATE:

            return {
                "success": False,
                "error":
                    f"Expected {SAMPLE_RATE} Hz audio, "
                    f"received {sr} Hz."
            }


        noisy = np.asarray(
            noisy,
            dtype=np.float32
        )


        # ====================================================
        # AUDIO INFORMATION
        # ====================================================

        num_samples = len(noisy)

        duration = (
            num_samples /
            SAMPLE_RATE
        )


        if num_samples == 0:

            return {
                "success": False,
                "error": "Audio file is empty."
            }


        # ====================================================
        # INPUT METRICS
        # ====================================================

        input_rms = float(
            np.sqrt(
                np.mean(
                    noisy ** 2
                )
            )
        )

        input_peak = float(
            np.max(
                np.abs(noisy)
            )
        )


        # ====================================================
        # TORCH TENSOR
        # ====================================================

        x = torch.from_numpy(
            noisy
        ).float()

        x = x.to(
            DEVICE
        ).unsqueeze(0)


        # ====================================================
        # HANN WINDOW
        # ====================================================

        window = torch.hann_window(
            N_FFT,
            device=DEVICE
        )


        # ====================================================
        # MASKNET INFERENCE
        # ====================================================

        inference_start = (
            time.perf_counter()
        )


        with torch.no_grad():

            # ----------------------------------------------
            # STFT
            # ----------------------------------------------

            X = torch.stft(

                x,

                n_fft=N_FFT,

                hop_length=HOP,

                window=window,

                return_complex=True
            )


            # ----------------------------------------------
            # MAGNITUDE
            # ----------------------------------------------

            magnitude = torch.abs(X)


            # ----------------------------------------------
            # MASKNET
            # ----------------------------------------------

            mask = model(
                magnitude
            )


            # ----------------------------------------------
            # APPLY MASK
            # ----------------------------------------------

            enhanced_spec = (
                X * mask
            )


            # ----------------------------------------------
            # ISTFT
            # ----------------------------------------------

            enhanced = torch.istft(

                enhanced_spec,

                n_fft=N_FFT,

                hop_length=HOP,

                window=window,

                length=num_samples
            )


        inference_time = (
            time.perf_counter()
            - inference_start
        )


        # ====================================================
        # BACK TO NUMPY
        # ====================================================

        enhanced = (
            enhanced
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


        # ====================================================
        # OUTPUT METRICS
        # ====================================================

        output_rms = float(
            np.sqrt(
                np.mean(
                    enhanced ** 2
                )
            )
        )

        output_peak = float(
            np.max(
                np.abs(enhanced)
            )
        )


        # ====================================================
        # ENCODE WAV
        # ====================================================

        output_buffer = io.BytesIO()

        sf.write(

            output_buffer,

            enhanced,

            SAMPLE_RATE,

            format="WAV",

            subtype="PCM_16"
        )


        enhanced_base64 = (
            base64.b64encode(
                output_buffer.getvalue()
            ).decode("utf-8")
        )


        # ====================================================
        # TIMING
        # ====================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        processing_time_ms = (
            total_time * 1000
        )

        inference_time_ms = (
            inference_time * 1000
        )

        real_time_factor = (
            inference_time /
            duration
            if duration > 0
            else 0
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success": True,

            "enhancedAudioBase64":
                enhanced_base64,

            "enhancedAudioFormat":
                "audio/wav",

            "metrics": {

                # No clean reference was supplied,
                # so these are intentionally unavailable.

                "inputSnr": None,

                "outputSnr": None,

                "snrImprovement": None,

                "siSdr": None,

                "stoi": None,

                "pesq": None,

                "processingTimeMs":
                    round(
                        processing_time_ms,
                        2
                    ),

                "inferenceTimeMs":
                    round(
                        inference_time_ms,
                        2
                    ),

                "realTimeFactor":
                    round(
                        real_time_factor,
                        3
                    ),

                "hasReference":
                    False,

                "inputRms":
                    input_rms,

                "outputRms":
                    output_rms,

                "metricNotes": {

                    "notice":
                        "Reference-based metrics require aligned clean audio."

                }

            },

            "modelInfo": {

                "modelName":
                    "MaskNet",

                "version":
                    "1.0",

                "architecture":
                    "Neural Spectral Masking",

                "sampleRate":
                    SAMPLE_RATE,

                "fftSize":
                    N_FFT,

                "hopSize":
                    HOP,

                "device":
                    DEVICE

            }

        }


    except Exception as e:

        print()
        print("MASKNET INFERENCE ERROR:")
        print(e)
        print()

        return {

            "success": False,

            "error":
                f"MaskNet inference failed: {str(e)}"

        }