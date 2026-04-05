from __future__ import annotations
from scipy.signal import butter, sosfilt, sosfilt_zi
import numpy as np
from ..methodclass import Method, RunOutput


class AS_BMFLC(Method):
    """
    Adaptive Sliding Bandlimited Multiple Fourier Linear Combiner (ASBMFLC)
    for adaptive tremor estimation.
    Uses a WFLC to estimate the instantaneous dominant frequency of the tremor,
    and then applies a BMFLC with a sliding window of basis functions centered around that frequency.
    Raw signal is first bandpassed by a 4th order butterworth filter before being fed to the WFLC.

    Based on the algorithm from:
    Wang, S. and Gao, Y. and Zhao, J. and Cai, H.,
    "Adaptive sliding bandlimited multiple fourier linear combiner for estimation of pathological tremor",
    Biomedical Signal Processing and Control,
    Volume 10,
    2014,
    Pages 260-274,
    ISSN 1746-8094,
    https://doi.org/10.1016/j.bspc.2013.10.004.

    Parameters:
    ----------
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of harmonics from sliding bandwidth to take in truncation.
        Bandwith endpoints are included, so n must be greater than 1.
    f0 : float
        Initial fundamental frequency in Hz for the WFLC to track. Should be within the bandwidth.
    mu_wflc : float
        Learning rate for the WFLC. Positive value.
    mu_0_wflc : float
        Learning rate for frequency estimate in the WFLC. Positive value.
    mu_bmflc : float
        Learning rate for the BMFLC. Positive value.
    bandwidth : tuple[float, float]
        Frequency range for bandlimited basis functions in Hz. Ex: (4, 12) Hz.
    window_length : float
        Length of the sliding window in Hz for the BMFLC basis functions.
    """

    def __init__(self,
                 fs: float,
                 n: int,
                 f0: float,
                 mu_wflc: float,
                 mu_0_wflc: float,
                 mu_bmflc: float,
                 bandwidth: tuple[float, float],
                 window_length: float):
        self.fs = fs
        self.n = n
        self.w0 = 2 * np.pi * f0
        self.mu_wflc = mu_wflc
        self.mu_0_wflc = mu_0_wflc
        self.mu_bmflc = mu_bmflc
        self.wmin, self.wmax = 2 * np.pi * np.array(bandwidth)
        self.L = 2 * np.pi * window_length

        self.dt = 1 / self.fs
        self.w0_sum = self.w0

        self.filter = butter(
            N=4,
            Wn=bandwidth,
            fs=self.fs,
            btype='bandpass',
            output='sos'
        )

        self.w_wflc = np.zeros(2 * n)
        self.w_bmflc = np.zeros(2 * (2 * n + 1) + 1)

        self.w_min, self.w_max = 2 * np.pi * np.array(bandwidth)

    def run(self, signal: np.ndarray) -> RunOutput:
        tremor_estimates, voluntary_estimates = self._estimate_components(signal)  # noqa: E501
        motion_estimates = voluntary_estimates + tremor_estimates
        return RunOutput(
            tremor_estimates=tremor_estimates,
            voluntary_estimates=voluntary_estimates,
            motion_estimates=motion_estimates
        )

    def _estimate_components(self, signal: np.ndarray) -> tuple[np.ndarray,
                                                                np.ndarray]:
        tremor_estimates = np.zeros_like(signal)
        voluntary_estimates = np.zeros_like(signal)
        zi = sosfilt_zi(self.filter) * signal[0]
        r = np.arange(self.n) + 1
        w_dom = (self.w_min + self.w_max) / 2

        for k, y in enumerate(signal):
            # Bandpass filtering
            s, zi = sosfilt(self.filter, [y], zi=zi)
            s = s[0]

            # WFLC frequency estimation
            x_wflc = np.concatenate([
                np.sin(self.w0_sum * r * self.dt),
                np.cos(self.w0_sum * r * self.dt)
            ])
            s_hat_wflc = self.w_wflc.T @ x_wflc
            error_wflc = s - s_hat_wflc
            self.w_wflc += 2 * self.mu_wflc * error_wflc * x_wflc
            self.w0 += 2 * self.mu_0_wflc * error_wflc * sum(
                r * (self.w_wflc[r - 1] * x_wflc[r + self.n - 1] -
                     self.w_wflc[r + self.n - 1] * x_wflc[r - 1])
            )
            self.w0_sum += self.w0

            # Update dominant frequency
            if (w_dom < self.w_max - self.L / 2 and
                    w_dom > self.w_min + self.L / 2):
                w_dom = self.w0
            else:
                w_dom = np.clip(
                    a=w_dom,
                    a_min=self.w_min + self.L / 2,
                    a_max=self.w_max - self.L / 2
                )

            # Update BMFLC sliding window of basis frequencies
            w0 = w_dom - (self.L / 2)
            w1 = w_dom + (self.L / 2)
            ws = w0 + (w1 - w0) * np.arange(2 * self.n + 1) / (2 * self.n)

            # BMFLC amplitude estimation
            x_bmflc = np.concatenate([
                np.sin(ws * k * self.dt),
                np.cos(ws * k * self.dt),
                [1.0]
            ])
            y_hat_bmflc = self.w_bmflc.T @ x_bmflc
            error_bmflc = y - y_hat_bmflc
            self.w_bmflc += 2 * self.mu_bmflc * error_bmflc * x_bmflc

            voluntary_estimates[k] = self.w_bmflc[-1]
            tremor_estimates[k] = y_hat_bmflc - self.w_bmflc[-1]

        return tremor_estimates, voluntary_estimates
