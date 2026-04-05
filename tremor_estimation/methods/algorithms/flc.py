from __future__ import annotations
from matplotlib.pylab import zeros_like
import numpy as np
from ..methodclass import Method, RunOutput


class FLC(Method):
    """
    Fourier Linear Combiner (FLC) for adaptive tremor estimation.

    Implements the algorithm from:
    Vaz, C. and Thakor, N.V.
    "Adaptive Fourier estimation of time-varying evoked potentials."
    in IEEE Transactions on Biomedical Engineering,
    vol. 36, no. 4, pp. 448-455, April 1989,
    doi: 10.1109/10.18751

    A real-time, zero-phase lag adaptive algorithm for
    estimating signal components at a known frequency using
    truncated Fourier basis functions with LMS-defined weights.
    In a signal composed of high-frequency tremor and low-frequency
    voluntary motion, FLC becomes an estimator of voluntary motion.

    Parameters:
    ----------
    fs : float
        Sampling frequency in Hz.
    f0 : float
        Fundamental frequency in Hz.
    n : int
        Number of harmonics to include in truncation.
    mu : float
        Learning rate for LMS adaptation. Positive value << 1/2.
    """

    def __init__(self,
                 fs: float,
                 f0: float,
                 n: int,
                 mu: float):
        self.fs = fs
        self.f0 = f0
        self.w0 = 2 * np.pi * self.f0
        self.n = n
        self.mu = mu

        self.dt = 1 / self.fs
        self.w = np.zeros(2 * self.n)

    def run(self, signal: np.ndarray) -> RunOutput:
        mean = np.mean(signal)
        s = signal - mean  # AC coupling
        voluntary_estimates = self._estimate_voluntary(s)
        tremor_estimates = s - voluntary_estimates
        motion_estimates = voluntary_estimates + tremor_estimates + mean
        return RunOutput(
            tremor_estimates=tremor_estimates,
            voluntary_estimates=voluntary_estimates,
            motion_estimates=motion_estimates
        )

    def _estimate_voluntary(self, signal: np.ndarray) -> np.ndarray:
        voluntary_estimates = zeros_like(signal)
        r = np.arange(self.n) + 1
        for k, y in enumerate(signal):
            x = np.concatenate([
                np.sin(self.w0 * r * k * self.dt),
                np.cos(self.w0 * r * k * self.dt)
            ])
            y_hat = self.w.T @ x
            voluntary_estimates[k] = y_hat
            error = y - y_hat
            self.w += 2 * self.mu * error * x
        return voluntary_estimates
