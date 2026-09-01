import numpy as np


class FxLMS:
    """Filtered-x LMS adaptive noise canceller.

    x(n): reference (noise) signal
    d(n): primary signal at the ear (noise that reached the listener)
    s_hat: estimated impulse response of the secondary path (speaker->air->error mic)
    """

    def __init__(self, n_taps=128, mu=0.01, secondary_path_estimate=None):
        self.n_taps = n_taps
        self.mu = mu
        self.w = np.zeros(n_taps, dtype=np.float64)
        self.x_buf = np.zeros(n_taps, dtype=np.float64)

        if secondary_path_estimate is None:
            secondary_path_estimate = np.array([1.0])  # ideal/no coloration
        self.s_hat = secondary_path_estimate
        self.xf_buf = np.zeros(n_taps, dtype=np.float64)  # filtered-x history

    def _filter_through_secondary_path(self, x_scalar):
        # push new sample through the estimated secondary path (simple FIR conv, streaming)
        self.xf_buf = np.roll(self.xf_buf, 1)
        self.xf_buf[0] = x_scalar
        s = self.s_hat
        n = min(len(s), len(self.xf_buf))
        return float(np.dot(s[:n], self.xf_buf[:n]))

    def step(self, x_n, d_n):
        """Process one sample. Returns (anti_noise_sample, error_sample)."""
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = x_n

        y_n = float(np.dot(self.w, self.x_buf))  # anti-noise output

        # after being played, anti-noise travels through secondary path -> y'(n)
        y_prime = self._filter_through_secondary_path(y_n)

        e_n = d_n + y_prime  # error mic picks up leftover noise + anti-noise

        x_filtered = self._filter_through_secondary_path(x_n)
        self.w += self.mu * e_n * self.x_buf  # simplified update using filtered-x sample

        return y_n, e_n

    def process_block(self, x_block, d_block):
        y_out = np.zeros_like(x_block)
        e_out = np.zeros_like(x_block)
        for i in range(len(x_block)):
            y_out[i], e_out[i] = self.step(x_block[i], d_block[i])
        return y_out, e_out


def estimate_secondary_path(play_fn, record_fn, sr=16000, n_taps=128, probe_len=2048):
    """Play a known probe signal, record via error mic, deconvolve to get S(z).
    play_fn(signal)->None, record_fn(n_samples)->np.ndarray, both real hardware I/O.
    On real hardware this replaces the placeholder identity estimate above.
    """
    rng = np.random.default_rng(0)
    probe = rng.standard_normal(probe_len).astype(np.float32) * 0.05
    play_fn(probe)
    recorded = record_fn(probe_len)
    # crude least-squares FIR estimate of S(z)
    X = np.zeros((probe_len - n_taps, n_taps))
    for i in range(n_taps):
        X[:, i] = probe[n_taps - i - 1: probe_len - i - 1]
    y = recorded[n_taps: probe_len]
    s_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    return s_hat


if __name__ == "__main__":
    # quick synthetic sanity check: cancel a 100Hz tone (rotor-like periodic noise)
    sr = 16000
    t = np.arange(sr * 2) / sr
    noise = 0.5 * np.sin(2 * np.pi * 100 * t)
    canceller = FxLMS(n_taps=64, mu=0.02)
    _, e = canceller.process_block(noise, noise)
    print("initial error power:", np.mean(noise[:500] ** 2))
    print("final error power:  ", np.mean(e[-500:] ** 2))