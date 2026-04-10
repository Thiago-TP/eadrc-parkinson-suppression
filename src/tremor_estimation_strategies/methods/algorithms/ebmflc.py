from __future__ import annotations
import numpy as np
from ..methodclass import Method, RunOutput


class EBMFLC(Method):
    """
    Enhance Bandlimited Multiple Fourier Linear Combiner (EBMFLC)
    for adaptive tremor estimation.

    Implements the algorithm from:
    Atashzar, S.F. and Shahbazi, M. and Samotus, O. and Tavakoli M. and Jog, M.S. and Patel, R.V.,
    "Characterization of Upper-Limb Pathological Tremors: Application to Design of an Augmented Haptic Rehabilitation System",
    in IEEE Journal of Selected Topics in Signal Processing,
    vol. 10, no. 5, pp. 888-903, Aug. 2016,
    doi: 10.1109/JSTSP.2016.2530632

    Modifies BMFLC with improved error calculation and windowed memory
    for simultaneous estimation of tremor and voluntary movement.

    Parameters:
    ----------
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of harmonics from total bandwidth to take in truncation.
        Bandwith endpoints are included, so n must be greater than 1.
    mu : float
        Learning rate for LMS adaptation. Positive value.
    total_bandwidth : tuple[float, float]
        The entire frequency range in Hz considered for basis functions
        when estimating the original signal. Ex: (0, 20) Hz.
    voluntary_bandwidth : tuple[float, float]
        The voluntary motion frequency range in Hz considered for basis
        functions when estimating voluntary motion. Ex: (0, 2) Hz.
    tremor_bandwidth : tuple[float, float]
        The tremor frequency range in Hz considered for basis functions
        when estimating tremor. Ex: (4, 12) Hz.
    window_time : float
        Time in seconds for windowed memory.
    minimum_impact : float
        Minimum impact a sample in the windowed memory can have.
        Must be a value between 0 and 1, where 0 means no minimum impact and
        1 means all samples in the window have equal impact regardless of age.
    """

    def __init__(self,
                 fs: float,
                 n: int,
                 mu: float,
                 total_bandwidth: tuple[float, float],
                 voluntary_bandwidth: tuple[float, float],
                 tremor_bandwidth: tuple[float, float],
                 window_time: float,
                 minimum_impact: float):
        self.fs = fs
        self.n = n
        self.mu = mu

        self.dt = 1 / self.fs
        self.w_m = np.zeros(2 * self.n)

        f_min, f_max = total_bandwidth
        freqs = f_min + np.arange(n) * (f_max - f_min) / (n - 1)

        f_min_vol, f_max_vol = voluntary_bandwidth
        self.voluntary_indices = np.where(
            (f_min_vol <= freqs) &
            (freqs <= f_max_vol)
        )[0].tolist()

        f_min_trem, f_max_trem = tremor_bandwidth
        self.tremor_indices = np.where(
            (freqs >= f_min_trem) &
            (freqs <= f_max_trem)
        )[0].tolist()

        self.ws = 2 * np.pi * freqs

        self.d = self.fs * window_time
        self.p = max(0, min(1, minimum_impact)) ** (1 / self.d)

    def run(self, signal: np.ndarray) -> RunOutput:
        return RunOutput(*self._estimate_components(signal))

    def _estimate_components(self, signal: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]:
        motion_estimates = np.zeros_like(signal)
        voluntary_estimates = np.zeros_like(signal)
        tremor_estimates = np.zeros_like(signal)

        for k, m in enumerate(signal):

            # Harmonics of total motion, voluntary and tremor components
            x_m = np.concatenate([
                np.sin(self.ws * k * self.dt),
                np.cos(self.ws * k * self.dt)
            ])
            x_v = x_m[self.voluntary_indices * 2]
            x_t = x_m[self.tremor_indices * 2]

            # Estimate of motion
            m_hat = self.w_m.T @ x_m

            # Estimates of voluntary and tremor components
            # Indices are doubled (concatenated) to include both harmonics
            w_v = self.w_m[self.voluntary_indices * 2]
            w_t = self.w_m[self.tremor_indices * 2]
            v_hat = w_v.T @ x_v
            t_hat = w_t.T @ x_t

            # Update of estimates histories
            voluntary_estimates[k] = v_hat
            tremor_estimates[k] = t_hat
            motion_estimates[k] = m_hat

            # Motion weights update
            error_m = m - m_hat
            self.w_m = self.p * self.w_m + 2 * self.mu * error_m * x_m

        return (
            tremor_estimates,
            voluntary_estimates,
            motion_estimates
        )
