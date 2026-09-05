import base64
import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
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

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "checkpoints" / "masknet.pt"

# ============================================================
# OPTIONAL METRIC LIBRARIES
# ============================================================

try:
    from pystoi import stoi

    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("WARNING: pystoi is not installed. STOI will be unavailable.")

try:
    from pesq import pesq

    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print(
        "WARNING: pesq is not installed. "
        "PESQ will be unavailable."
    )

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ANC MaskNet Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DEVICE & MODEL
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "=" * 60)
print("Loading MaskNet")
print("=" * 60)
print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"MaskNet checkpoint not found: {CHECKPOINT}"
    )

model = MaskNet(
    n_freq=N_FFT // 2 + 1
).to(DEVICE)

try:
    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )
except TypeError:
    # Compatibility with older PyTorch versions
    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

model.load_state_dict(state_dict)
model.eval()

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
# AUDIO HELPERS
# ============================================================


def decode_base64_audio(encoded: str) -> tuple[np.ndarray, int]:
    """Decode base64 WAV/audio bytes into mono float32 samples."""

    try:
        audio_bytes = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError(
            f"Invalid base64 audio: {exc}"
        ) from exc

    try:
        audio, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
        )
    except Exception as exc:
        raise ValueError(
            f"Could not decode audio: {exc}"
        ) from exc

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1,
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    return audio, sr


def rms(samples: np.ndarray) -> float:
    """Calculate RMS amplitude."""

    if len(samples) == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.square(samples)
            )
        )
    )


def align_signals(
    reference: np.ndarray,
    degraded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trim both signals to the same length.

    The UI preprocessing produces 16 kHz mono audio, so no resampling
    is necessary here.
    """

    length = min(
        len(reference),
        len(degraded),
    )

    if length == 0:
        raise ValueError(
            "Reference and degraded audio must not be empty."
        )

    reference = reference[:length]
    degraded = degraded[:length]

    return reference, degraded


# ============================================================
# METRIC CALCULATIONS
# ============================================================


def calculate_snr(
    clean: np.ndarray,
    degraded: np.ndarray,
) -> float:
    """
    Calculate SNR using:

        SNR = 10 * log10(clean_energy / noise_energy)

    where:

        noise = degraded - clean
    """

    clean, degraded = align_signals(
        clean,
        degraded,
    )

    noise = degraded - clean

    clean_energy = np.mean(
        np.square(clean)
    )

    noise_energy = np.mean(
        np.square(noise)
    )

    if clean_energy <= 1e-12:
        return float("-inf")

    if noise_energy <= 1e-12:
        return 100.0

    return float(
        10.0
        * np.log10(
            clean_energy
            / noise_energy
        )
    )


def calculate_si_sdr(
    clean: np.ndarray,
    estimated: np.ndarray,
) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio.

    SI-SDR:

        s_target = alpha * clean

        alpha = <estimated, clean> / ||clean||²

        SI-SDR = 10 log10(
            ||s_target||² /
            ||estimated - s_target||²
        )
    """

    clean, estimated = align_signals(
        clean,
        estimated,
    )

    # Remove DC offset
    clean = clean - np.mean(clean)
    estimated = estimated - np.mean(estimated)

    clean_energy = np.sum(
        np.square(clean)
    )

    if clean_energy <= 1e-12:
        return float("-inf")

    alpha = (
        np.sum(
            estimated * clean
        )
        / clean_energy
    )

    target = alpha * clean
    distortion = estimated - target

    target_energy = np.sum(
        np.square(target)
    )

    distortion_energy = np.sum(
        np.square(distortion)
    )

    if distortion_energy <= 1e-12:
        return 100.0

    return float(
        10.0
        * np.log10(
            target_energy
            / distortion_energy
        )
    )


def calculate_reference_metrics(
    noisy: np.ndarray,
    enhanced: np.ndarray,
    clean: np.ndarray,
) -> dict:
    """
    Calculate full-reference objective metrics.
    """

    # Align noisy with clean
    clean_for_input, noisy_for_input = align_signals(
        clean,
        noisy,
    )

    # Align enhanced with clean
    clean_for_output, enhanced_for_output = align_signals(
        clean,
        enhanced,
    )

    # Use common length for all metrics
    common_length = min(
        len(clean_for_input),
        len(clean_for_output),
    )

    clean_common = clean[:common_length]
    noisy_common = noisy[:common_length]
    enhanced_common = enhanced[:common_length]

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    input_snr = calculate_snr(
        clean_common,
        noisy_common,
    )

    output_snr = calculate_snr(
        clean_common,
        enhanced_common,
    )

    snr_improvement = (
        output_snr - input_snr
    )

    # --------------------------------------------------------
    # SI-SDR
    # --------------------------------------------------------

    si_sdr = calculate_si_sdr(
        clean_common,
        enhanced_common,
    )

    # --------------------------------------------------------
    # STOI
    # --------------------------------------------------------

    stoi_score = None
    stoi_error = None

    if STOI_AVAILABLE:
        try:
            stoi_score = float(
                stoi(
                    clean_common,
                    enhanced_common,
                    SAMPLE_RATE,
                    extended=False,
                )
            )
        except Exception as exc:
            stoi_error = str(exc)

    # --------------------------------------------------------
    # PESQ
    # --------------------------------------------------------

    pesq_score = None
    pesq_error = None

    if PESQ_AVAILABLE:
        try:
            # Wideband PESQ at 16 kHz
            pesq_score = float(
                pesq(
                    SAMPLE_RATE,
                    clean_common,
                    enhanced_common,
                    "wb",
                )
            )
        except Exception as exc:
            pesq_error = str(exc)

    # --------------------------------------------------------
    # Notes
    # --------------------------------------------------------

    notes = {
        "methodology": (
            "Full-reference evaluation using aligned clean audio. "
            "SNR and SI-SDR are calculated directly. "
            "STOI uses pystoi. PESQ uses the PESQ implementation "
            "when installed."
        )
    }

    if not STOI_AVAILABLE:
        notes["stoi"] = (
            "STOI unavailable because pystoi is not installed."
        )

    elif stoi_error:
        notes["stoi"] = (
            f"STOI calculation failed: {stoi_error}"
        )

    if not PESQ_AVAILABLE:
        notes["pesq"] = (
            "PESQ unavailable because the pesq package is not installed."
        )

    elif pesq_error:
        notes["pesq"] = (
            f"PESQ calculation failed: {pesq_error}"
        )

    return {
        "inputSnr": round(input_snr, 2),
        "outputSnr": round(output_snr, 2),
        "snrImprovement": round(
            snr_improvement,
            2,
        ),
        "siSdr": round(
            si_sdr,
            2,
        ),
        "stoi": (
            round(stoi_score, 4)
            if stoi_score is not None
            else None
        ),
        "pesq": (
            round(pesq_score, 3)
            if pesq_score is not None
            else None
        ),
        "hasReference": True,
        "metricNotes": notes,
    }


# ============================================================
# STATUS ENDPOINT
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
        "checkpoint": str(CHECKPOINT),
        "stoiAvailable": STOI_AVAILABLE,
        "pesqAvailable": PESQ_AVAILABLE,
    }


# ============================================================
# PROCESS ENDPOINT
# ============================================================


@app.post("/api/anc/process")
async def process_audio(
    req: AncProcessRequest,
):
    total_start = time.perf_counter()

    try:
        # ----------------------------------------------------
        # 1. Decode noisy audio
        # ----------------------------------------------------

        noisy, sr = decode_base64_audio(
            req.audioBase64
        )

        if sr != SAMPLE_RATE:
            return {
                "success": False,
                "error": (
                    f"Expected {SAMPLE_RATE} Hz audio, "
                    f"received {sr} Hz."
                ),
            }

        if len(noisy) == 0:
            return {
                "success": False,
                "error": "Audio file is empty.",
            }

        duration = (
            len(noisy)
            / SAMPLE_RATE
        )

        input_rms = rms(noisy)

        # ----------------------------------------------------
        # 2. Convert to PyTorch
        # ----------------------------------------------------

        x = (
            torch.from_numpy(noisy)
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )

        window = torch.hann_window(
            N_FFT,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # 3. MaskNet inference
        # ----------------------------------------------------

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

            mask = model(
                magnitude
            )

            enhanced_spec = (
                X * mask
            )

            enhanced = torch.istft(
                enhanced_spec,
                n_fft=N_FFT,
                hop_length=HOP,
                window=window,
                length=len(noisy),
            )

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        enhanced = (
            enhanced
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        output_rms = rms(
            enhanced
        )

        # ----------------------------------------------------
        # 4. Calculate basic timing
        # ----------------------------------------------------

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
            inference_time
            / duration
            if duration > 0
            else 0.0
        )

        # ----------------------------------------------------
        # 5. Calculate reference metrics
        # ----------------------------------------------------

        reference_metrics = {
            "inputSnr": None,
            "outputSnr": None,
            "snrImprovement": None,
            "siSdr": None,
            "stoi": None,
            "pesq": None,
            "hasReference": False,
            "metricNotes": {
                "referenceRequired": (
                    "Upload an aligned clean reference "
                    "audio file to calculate full-reference "
                    "metrics."
                )
            },
        }

        if req.referenceAudioBase64:

            clean, reference_sr = (
                decode_base64_audio(
                    req.referenceAudioBase64
                )
            )

            if reference_sr != SAMPLE_RATE:
                return {
                    "success": False,
                    "error": (
                        f"Expected clean reference at "
                        f"{SAMPLE_RATE} Hz, received "
                        f"{reference_sr} Hz."
                    ),
                }

            if len(clean) == 0:
                return {
                    "success": False,
                    "error": (
                        "Clean reference audio is empty."
                    ),
                }

            reference_metrics = (
                calculate_reference_metrics(
                    noisy,
                    enhanced,
                    clean,
                )
            )

        # ----------------------------------------------------
        # 6. Encode enhanced WAV
        # ----------------------------------------------------

        output_buffer = io.BytesIO()

        sf.write(
            output_buffer,
            enhanced,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

        enhanced_base64 = (
            base64.b64encode(
                output_buffer.getvalue()
            ).decode("utf-8")
        )

        # ----------------------------------------------------
        # 7. Final response
        # ----------------------------------------------------

        metrics = {
            **reference_metrics,
            "processingTimeMs": round(
                processing_time_ms,
                2,
            ),
            "inferenceTimeMs": round(
                inference_time_ms,
                2,
            ),
            "realTimeFactor": round(
                real_time_factor,
                3,
            ),
            "inputRms": round(
                input_rms,
                6,
            ),
            "outputRms": round(
                output_rms,
                6,
            ),
        }

        return {
            "success": True,

            "enhancedAudioBase64": (
                enhanced_base64
            ),

            "enhancedAudioFormat": (
                "audio/wav"
            ),

            "metrics": metrics,

            "modelInfo": {
                "modelName": "MaskNet",
                "version": "1.0",
                "architecture": (
                    "Neural Spectral Masking"
                ),
                "sampleRate": SAMPLE_RATE,
                "fftSize": N_FFT,
                "hopSize": HOP,
                "device": DEVICE,
            },
        }

    except Exception as exc:

        print(
            "\nMASKNET INFERENCE ERROR:"
        )
        print(exc)
        print()

        return {
            "success": False,
            "error": (
                f"MaskNet inference failed: {exc}"
            ),
        }