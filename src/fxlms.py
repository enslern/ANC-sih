import numpy as np


class FxLMS:
    """
    Real-time friendly Filtered-x LMS adaptive filter.

    IMPORTANT:
    x = reference/noise signal
    d = desired signal containing the noise to be cancelled

    Do NOT use the same speech-containing signal for both x and d
    in a real microphone ANC system.
    """

    def __init__(
        self,
        n_taps=64,
        mu=0.0001,
        secondary_path_estimate=None
    ):
        self.n_taps = n_taps
        self.mu = mu

        # Adaptive filter coefficients
        self.w = np.zeros(n_taps, dtype=np.float32)

        # Reference signal buffer
        self.x_buf = np.zeros(n_taps, dtype=np.float32)
        self.x_pos = 0

        # Filtered-x buffer
        self.xf_buf = np.zeros(n_taps, dtype=np.float32)
        self.xf_pos = 0

        # Secondary path
        if secondary_path_estimate is None:
            secondary_path_estimate = np.array(
                [1.0],
                dtype=np.float32
            )
        else:
            secondary_path_estimate = np.asarray(
                secondary_path_estimate,
                dtype=np.float32
            )

        self.s_hat = secondary_path_estimate

        # Separate buffers:
        # one for anti-noise
        # one for reference
        self.y_sec_buf = np.zeros(
            len(self.s_hat),
            dtype=np.float32
        )

        self.x_sec_buf = np.zeros(
            len(self.s_hat),
            dtype=np.float32
        )

        self.y_sec_pos = 0
        self.x_sec_pos = 0

        self.indices = np.arange(
            n_taps,
            dtype=np.int32
        )

    def _secondary_filter_y(self, value):
        """Filter anti-noise through secondary path."""

        n = len(self.s_hat)

        self.y_sec_buf[self.y_sec_pos] = value

        indices = (
            self.y_sec_pos
            - np.arange(n)
        ) % n

        result = np.dot(
            self.s_hat,
            self.y_sec_buf[indices]
        )

        self.y_sec_pos += 1

        if self.y_sec_pos >= n:
            self.y_sec_pos = 0

        return float(result)

    def _secondary_filter_x(self, value):
        """Filter reference through secondary path."""

        n = len(self.s_hat)

        self.x_sec_buf[self.x_sec_pos] = value

        indices = (
            self.x_sec_pos
            - np.arange(n)
        ) % n

        result = np.dot(
            self.s_hat,
            self.x_sec_buf[indices]
        )

        self.x_sec_pos += 1

        if self.x_sec_pos >= n:
            self.x_sec_pos = 0

        return float(result)

    def step(self, x_n, d_n):
        """
        Process one sample.

        x_n:
            reference/noise signal

        d_n:
            desired signal containing the noise

        Returns:
            y_n = anti-noise
            e_n = residual
        """

        # -----------------------------------------------------
        # 1. Update reference buffer
        # -----------------------------------------------------

        self.x_buf[self.x_pos] = x_n

        indices = (
            self.x_pos
            - self.indices
        ) % self.n_taps

        # -----------------------------------------------------
        # 2. Generate anti-noise
        # -----------------------------------------------------

        y_n = np.dot(
            self.w,
            self.x_buf[indices]
        )

        # -----------------------------------------------------
        # 3. Secondary path
        # -----------------------------------------------------

        y_prime = self._secondary_filter_y(
            y_n
        )

        # -----------------------------------------------------
        # 4. Error
        # -----------------------------------------------------

        e_n = d_n - y_prime

        # -----------------------------------------------------
        # 5. Filter reference through secondary path
        # -----------------------------------------------------

        x_filtered = self._secondary_filter_x(
            x_n
        )

        self.xf_buf[self.xf_pos] = x_filtered

        xf_indices = (
            self.xf_pos
            - self.indices
        ) % self.n_taps

        # -----------------------------------------------------
        # 6. Adaptation
        # -----------------------------------------------------

        x_energy = np.mean(
            self.x_buf[indices]
            ** 2
        )

        if x_energy > 1e-5 and self.mu > 0:

            self.w += (
                self.mu
                * e_n
                * self.xf_buf[xf_indices]
            )

        # Prevent unstable coefficients
        np.clip(
            self.w,
            -2.0,
            2.0,
            out=self.w
        )

        # Move circular buffers
        self.x_pos -= 1

        if self.x_pos < 0:
            self.x_pos = self.n_taps - 1

        self.xf_pos -= 1

        if self.xf_pos < 0:
            self.xf_pos = self.n_taps - 1

        return float(y_n), float(e_n)

    def process_block(self, x_block, d_block):

        x_block = np.asarray(
            x_block,
            dtype=np.float32
        )

        d_block = np.asarray(
            d_block,
            dtype=np.float32
        )

        y_out = np.empty_like(
            x_block
        )

        e_out = np.empty_like(
            d_block
        )

        for i in range(len(x_block)):

            y_out[i], e_out[i] = self.step(
                x_block[i],
                d_block[i]
            )

        return y_out, e_out


def estimate_secondary_path(
    play_fn,
    record_fn,
    sr=16000,
    n_taps=128,
    probe_len=2048
):
    """
    Estimate the speaker -> microphone secondary path.
    """

    rng = np.random.default_rng(0)

    probe = (
        rng.standard_normal(probe_len)
        .astype(np.float32)
        * 0.05
    )

    play_fn(probe)

    recorded = record_fn(
        probe_len
    )

    recorded = np.asarray(
        recorded,
        dtype=np.float32
    )

    X = np.zeros(
        (
            probe_len - n_taps,
            n_taps
        ),
        dtype=np.float32
    )

    for i in range(n_taps):

        X[:, i] = probe[
            n_taps - i - 1:
            probe_len - i - 1
        ]

    y = recorded[
        n_taps:
        probe_len
    ]

    s_hat, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )

    return s_hat.astype(
        np.float32
    )


if __name__ == "__main__":

    sr = 16000

    t = np.arange(
        sr * 2
    ) / sr

    noise = (
        0.5
        * np.sin(
            2 * np.pi * 100 * t
        )
    ).astype(np.float32)

    canceller = FxLMS(
        n_taps=64,
        mu=0.0001
    )

    _, e = canceller.process_block(
        noise,
        noise
    )

    print(
        "Initial error power:",
        np.mean(
            noise[:500] ** 2
        )
    )

    print(
        "Final error power:",
        np.mean(
            e[-500:] ** 2
        )
    )