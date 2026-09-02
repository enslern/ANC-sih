import os
import glob
import random
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from audiomentations import Compose, AddGaussianNoise, ClippingDistortion

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SAMPLE_RATE, CLEAN_DIR, NOISE_DIR, GEN_DIR, SNR_RANGE_DB


def load_wav(path, sr=SAMPLE_RATE):
    audio, file_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio


def simulate_room_path(signal, sr=SAMPLE_RATE):
    room = pra.ShoeBox(
        [6, 5, 3],
        fs=sr,
        materials=pra.Material(0.3),
        max_order=8,
    )
    room.add_source([1.0, 1.0, 1.5], signal=signal)
    mic_loc = np.array([[3.0], [3.0], [1.5]])
    room.add_microphone_array(pra.MicrophoneArray(mic_loc, room.fs))
    room.simulate()
    out = room.mic_array.signals[0, : len(signal)]
    return out.astype(np.float32)


def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    start = random.randint(0, len(noise) - len(clean))
    noise = noise[start:start + len(clean)]

    clean_power = np.mean(clean ** 2) + 1e-9
    noise_power = np.mean(noise ** 2) + 1e-9
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    mixed = clean + noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
        clean = clean / peak
    return mixed.astype(np.float32), clean.astype(np.float32)


augment = Compose([
    AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.003, p=0.3),
    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10, p=0.15),
])


def generate_dataset(n_examples=2000, seg_seconds=3.0, use_room_sim=True):
    os.makedirs(os.path.join(GEN_DIR, "noisy"), exist_ok=True)
    os.makedirs(os.path.join(GEN_DIR, "clean"), exist_ok=True)

    clean_files = glob.glob(os.path.join(CLEAN_DIR, "**", "*.wav"), recursive=True)
    noise_files = glob.glob(os.path.join(NOISE_DIR, "**", "*.wav"), recursive=True)

    if not clean_files or not noise_files:
        raise RuntimeError(
            f"Need wavs in {CLEAN_DIR} and {NOISE_DIR}. "
            "Populate with LibriSpeech (clean) and ESC-50 (noise)."
        )

    seg_len = int(seg_seconds * SAMPLE_RATE)

    for i in range(n_examples):
        clean_path = random.choice(clean_files)
        noise_path = random.choice(noise_files)

        clean = load_wav(clean_path)
        noise = load_wav(noise_path)

        if len(clean) < seg_len:
            clean = np.pad(clean, (0, seg_len - len(clean)))
        else:
            start = random.randint(0, len(clean) - seg_len)
            clean = clean[start:start + seg_len]

        if use_room_sim and random.random() < 0.5:
            try:
                clean_path_audio = simulate_room_path(clean)
                clean_path_audio = clean_path_audio[:seg_len]
                if len(clean_path_audio) == seg_len:
                    clean = clean_path_audio
            except Exception:
                pass

        snr_db = random.uniform(*SNR_RANGE_DB)
        noisy, clean_seg = mix_at_snr(clean, noise, snr_db)
        noisy = augment(samples=noisy, sample_rate=SAMPLE_RATE)

        sf.write(os.path.join(GEN_DIR, "noisy", f"{i:05d}.wav"), noisy, SAMPLE_RATE)
        sf.write(os.path.join(GEN_DIR, "clean", f"{i:05d}.wav"), clean_seg, SAMPLE_RATE)

        if i % 100 == 0:
            print(f"[{i}/{n_examples}] snr={snr_db:.1f}dB noise={os.path.basename(noise_path)}")

    print("Done. Wrote dataset to", GEN_DIR)


if __name__ == "__main__":
    generate_dataset()