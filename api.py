import base64
import io
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Resolve checkpoint relative to this api.py file.
# This is safer for deployment because the server does not
# depend on the directory from which uvicorn was started.
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "checkpoints" / "masknet.pt"


# ============================================================
# OPTIONAL METRIC LIBRARIES
# ============================================================

# STOI
try:
    from pystoi import stoi

    STOI_AVAILABLE = True

except ImportError:
    STOI_AVAILABLE = False
    print(
        "WARNING: pystoi is not installed. "
        "STOI will be unavailable."
    )


# PESQ
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
# FASTAPI & CORS
# ============================================================

app = FastAPI(
    title="ANC MaskNet Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,

    # Development + deployed frontend support
    allow_origins=["*"],

    # We don't use cookies/authentication for this API.
    # Keeping this False is correct when using "*".
    allow_credentials=False,

    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DEVICE & MODEL LOADING
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n" + "=" * 60)
print("Loading MaskNet")
print("=" * 60)

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"MaskNet checkpoint not found: {CHECKPOINT}"
    )


# Create model
model = MaskNet(
    n_freq=N_FFT // 2 + 1
).to(DEVICE)


# Load checkpoint
try:

    # Newer PyTorch versions
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


model.load_state_dict(
    state_dict
)

model.eval()


print("MaskNet loaded successfully.")
print("STOI available:", STOI_AVAILABLE)
print("PESQ available:", PESQ_AVAILABLE)
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

    # Optional clean/reference audio
    referenceAudioBase64: str | None = None


# ============================================================
# AUDIO HELPERS
# ============================================================


def decode_audio_base64(
    encoded_audio: str,
) -> tuple[np.ndarray, int]:
    """
    Decode base64 audio into a mono float32 NumPy array.

    The frontend sends a 16 kHz mono WAV, but this function
    also handles stereo audio by downmixing it to mono.
    """

    try:

        audio_bytes = base64.b64decode(
            encoded_audio
        )

    except Exception as e:

        raise ValueError(
            f"Invalid base64 audio: {e}"
        )


    try:

        audio, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
        )

    except Exception as e:

        raise ValueError(
            f"Could not decode audio: {e}"
        )


    # Stereo -> mono
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


def align_signals(
    clean: np.ndarray,
    degraded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trim two signals to their common length.

    This assumes the clean and degraded recordings represent
    the same audio content and are already approximately aligned.
    """

    length = min(
        len(clean),
        len(degraded),
    )


    if length <= 0:

        raise ValueError(
            "Audio signals are empty."
        )


    clean = clean[:length]
    degraded = degraded[:length]


    return clean, degraded


def calculate_rms(
    audio: np.ndarray,
) -> float:
    """
    Calculate RMS amplitude.
    """

    if len(audio) == 0:

        return 0.0


    return float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )


# ============================================================
# SNR
# ============================================================


def calculate_snr(
    clean: np.ndarray,
    degraded: np.ndarray,
) -> float:
    """
    Calculate SNR relative to a clean reference.

    SNR = 10 * log10(
        clean_power / noise_power
    )

    noise = degraded - clean
    """

    clean, degraded = align_signals(
        clean,
        degraded,
    )


    noise = (
        degraded - clean
    )


    clean_power = np.mean(
        np.square(clean)
    )

    noise_power = np.mean(
        np.square(noise)
    )


    if clean_power <= 1e-12:

        return float("-inf")


    if noise_power <= 1e-12:

        return 100.0


    snr = (
        10.0
        * np.log10(
            clean_power
            / noise_power
        )
    )


    return float(snr)


# ============================================================
# SI-SDR
# ============================================================


def calculate_si_sdr(
    clean: np.ndarray,
    estimated: np.ndarray,
) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio.

    SI-SDR is calculated using the standard projection of the
    estimated signal onto the clean reference.
    """

    clean, estimated = align_signals(
        clean,
        estimated,
    )


    # Remove DC offset
    clean = (
        clean
        - np.mean(clean)
    )

    estimated = (
        estimated
        - np.mean(estimated)
    )


    clean_energy = np.sum(
        np.square(clean)
    )


    if clean_energy <= 1e-12:

        return float("-inf")


    # Projection coefficient
    alpha = (
        np.sum(
            estimated * clean
        )
        / clean_energy
    )


    target = (
        alpha * clean
    )


    distortion = (
        estimated - target
    )


    target_energy = np.sum(
        np.square(target)
    )

    distortion_energy = np.sum(
        np.square(distortion)
    )


    if distortion_energy <= 1e-12:

        return 100.0


    si_sdr = (
        10.0
        * np.log10(
            target_energy
            / distortion_energy
        )
    )


    return float(si_sdr)


# ============================================================
# REFERENCE-BASED METRICS
# ============================================================


def calculate_reference_metrics(
    noisy: np.ndarray,
    enhanced: np.ndarray,
    clean: np.ndarray,
) -> dict:
    """
    Calculate reference-based audio quality metrics.

    Metrics:
        - Input SNR
        - Output SNR
        - SNR Improvement
        - SI-SDR
        - STOI
        - PESQ

    The clean reference and processed signals are trimmed to
    their common length before evaluation.
    """

    # --------------------------------------------------------
    # Align all three signals
    # --------------------------------------------------------

    common_length = min(
        len(noisy),
        len(enhanced),
        len(clean),
    )


    if common_length <= 0:

        raise ValueError(
            "Clean, noisy and enhanced audio must not be empty."
        )


    noisy = noisy[
        :common_length
    ]

    enhanced = enhanced[
        :common_length
    ]

    clean = clean[
        :common_length
    ]


    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    input_snr = calculate_snr(
        clean,
        noisy,
    )


    output_snr = calculate_snr(
        clean,
        enhanced,
    )


    snr_improvement = (
        output_snr
        - input_snr
    )


    # --------------------------------------------------------
    # SI-SDR
    # --------------------------------------------------------

    si_sdr = calculate_si_sdr(
        clean,
        enhanced,
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
                    clean,
                    enhanced,
                    SAMPLE_RATE,
                    extended=False,
                )
            )

        except Exception as e:

            stoi_error = str(e)


    # --------------------------------------------------------
    # PESQ
    # --------------------------------------------------------

    pesq_score = None
    pesq_error = None


    if PESQ_AVAILABLE:

        try:

            # 16 kHz = wideband PESQ
            pesq_score = float(
                pesq(
                    SAMPLE_RATE,
                    clean,
                    enhanced,
                    "wb",
                )
            )

        except Exception as e:

            pesq_error = str(e)


    # --------------------------------------------------------
    # Metric notes
    # --------------------------------------------------------

    metric_notes = {
        "methodology": (
            "Full-reference evaluation against the uploaded "
            "clean audio. Signals are trimmed to their common "
            "length before evaluation."
        )
    }


    if not STOI_AVAILABLE:

        metric_notes["stoi"] = (
            "STOI unavailable: install pystoi."
        )

    elif stoi_error:

        metric_notes["stoi"] = (
            f"STOI calculation failed: {stoi_error}"
        )


    if not PESQ_AVAILABLE:

        metric_notes["pesq"] = (
            "PESQ unavailable: install pesq."
        )

    elif pesq_error:

        metric_notes["pesq"] = (
            f"PESQ calculation failed: {pesq_error}"
        )


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "inputSnr": round(
            input_snr,
            2,
        ),

        "outputSnr": round(
            output_snr,
            2,
        ),

        "snrImprovement": round(
            snr_improvement,
            2,
        ),

        "siSdr": round(
            si_sdr,
            2,
        ),

        "stoi": (
            round(
                stoi_score,
                4,
            )
            if stoi_score is not None
            else None
        ),

        "pesq": (
            round(
                pesq_score,
                3,
            )
            if pesq_score is not None
            else None
        ),

        "hasReference": True,

        "metricNotes": metric_notes,
    }


# ============================================================
# STATUS ENDPOINT
# ============================================================


@app.get("/api/anc/status")
def health_check():

    return {

        "status": "online",

        "backend": (
            "Python FastAPI + PyTorch"
        ),

        "model": "MaskNet",

        "device": DEVICE,

        "sampleRate": SAMPLE_RATE,

        "fftSize": N_FFT,

        "hopSize": HOP,

        "checkpoint": str(
            CHECKPOINT
        ),

        "stoiAvailable": (
            STOI_AVAILABLE
        ),

        "pesqAvailable": (
            PESQ_AVAILABLE
        ),
    }


# ============================================================
# MAIN AUDIO PROCESSING ENDPOINT
# ============================================================


@app.post("/api/anc/process")
async def process_audio(
    req: AncProcessRequest,
):

    total_start = (
        time.perf_counter()
    )


    try:

        # ====================================================
        # 1. Decode noisy input audio
        # ====================================================

        noisy, sr = (
            decode_audio_base64(
                req.audioBase64
            )
        )


        # The frontend is expected to send 16 kHz audio.
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


        num_samples = len(
            noisy
        )


        duration = (
            num_samples
            / SAMPLE_RATE
        )


        input_rms = calculate_rms(
            noisy
        )


        input_peak = float(
            np.max(
                np.abs(noisy)
            )
        )


        # ====================================================
        # 2. Convert to PyTorch
        # ====================================================

        x = (
            torch
            .from_numpy(noisy)
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )


        window = torch.hann_window(
            N_FFT,
            device=DEVICE,
        )


        # ====================================================
        # 3. MaskNet inference
        # ====================================================

        inference_start = (
            time.perf_counter()
        )


        with torch.no_grad():

            # STFT
            X = torch.stft(
                x,

                n_fft=N_FFT,

                hop_length=HOP,

                window=window,

                return_complex=True,
            )


            # Magnitude spectrogram
            magnitude = torch.abs(
                X
            )


            # Neural mask
            mask = model(
                magnitude
            )


            # Apply mask
            enhanced_spec = (
                X * mask
            )


            # Inverse STFT
            enhanced = torch.istft(
                enhanced_spec,

                n_fft=N_FFT,

                hop_length=HOP,

                window=window,

                length=num_samples,
            )


        inference_time = (
            time.perf_counter()
            - inference_start
        )


        # ====================================================
        # 4. Convert output to NumPy
        # ====================================================

        enhanced = (
            enhanced
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


        output_rms = calculate_rms(
            enhanced
        )


        output_peak = float(
            np.max(
                np.abs(enhanced)
            )
        )


        # ====================================================
        # 5. Timing metrics
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
            inference_time / duration
            if duration > 0
            else 0
        )


        # ====================================================
        # 6. Default metrics
        #
        # Without a clean reference, the standardized
        # reference-based metrics cannot be calculated.
        # ====================================================

        reference_metrics = {

            "inputSnr": None,

            "outputSnr": None,

            "snrImprovement": None,

            "siSdr": None,

            "stoi": None,

            "pesq": None,

            "hasReference": False,

            "metricNotes": {

                "notice": (
                    "Upload a clean reference recording "
                    "of the same speech/content to calculate "
                    "SNR improvement, SI-SDR, STOI and PESQ."
                )
            },
        }


        # ====================================================
        # 7. Process clean reference if supplied
        # ====================================================

        if req.referenceAudioBase64:

            clean, reference_sr = (
                decode_audio_base64(
                    req.referenceAudioBase64
                )
            )


            # Reference must also be 16 kHz.
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


            # Calculate actual standardized metrics
            reference_metrics = (
                calculate_reference_metrics(
                    noisy=noisy,

                    enhanced=enhanced,

                    clean=clean,
                )
            )


        # ====================================================
        # 8. Encode enhanced audio as WAV
        # ====================================================

        output_buffer = (
            io.BytesIO()
        )


        sf.write(

            output_buffer,

            enhanced,

            SAMPLE_RATE,

            format="WAV",

            subtype="PCM_16",
        )


        enhanced_base64 = (
            base64
            .b64encode(
                output_buffer.getvalue()
            )
            .decode("utf-8")
        )


        # ====================================================
        # 9. Final metrics object
        # ====================================================

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

            "inputPeak": round(
                input_peak,
                6,
            ),

            "outputPeak": round(
                output_peak,
                6,
            ),
        }


        # ====================================================
        # 10. Return response
        # ====================================================

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


    except Exception as e:

        print(
            "\nMASKNET INFERENCE ERROR:"
        )

        print(e)

        print()


        return {

            "success": False,

            "error": (
                f"MaskNet inference failed: {str(e)}"
            ),
        }
