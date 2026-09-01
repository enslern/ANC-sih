import numpy as np
from pesq import pesq
from pystoi import stoi


def compute_snr(clean, enhanced):
    noise = clean - enhanced
    snr = 10 * np.log10((np.sum(clean ** 2) + 1e-9) / (np.sum(noise ** 2) + 1e-9))
    return snr


def compute_stoi(clean, enhanced, sr=16000):
    n = min(len(clean), len(enhanced))
    return stoi(clean[:n], enhanced[:n], sr, extended=False)


def compute_pesq(clean, enhanced, sr=16000):
    n = min(len(clean), len(enhanced))
    mode = "wb" if sr == 16000 else "nb"
    return pesq(sr, clean[:n], enhanced[:n], mode)


def evaluate(clean, enhanced, sr=16000):
    return {
        "SNR_dB": compute_snr(clean, enhanced),
        "STOI": compute_stoi(clean, enhanced, sr),
        "PESQ": compute_pesq(clean, enhanced, sr),
    }