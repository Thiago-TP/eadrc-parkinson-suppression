from __future__ import annotations
import numpy as np
from ..methodclass import Method, RunOutput


class BMFLC(Method):
    """
    Bandlimited Multiple Fourier Linear Combiner (BMFLC)
    for adaptive tremor estimation.
    Extends FLC with multiple bandlimited basis functions for
    improved selectivity in comparison with WFLC.

    Based on the algorithm from:
    K. C. Veluvolu, K.C and Tan, U.X. and Latt, W.T. and Shee, C.Y. and Ang, W.T.,
    "Bandlimited Multiple Fourier Linear Combiner for Real-time Tremor Compensation,"
    29th Annual International Conference of the IEEE Engineering in Medicine and Biology Society,
    Lyon, France, 2007, pp. 2847-2850,
    doi: 10.1109/IEMBS.2007.4352922

    The vector of harmonics was extended to include a constant term,
    which is useful for estimating the voluntary component of the signal.
    For details on the extension, read section 3.2 of the paper:

    Veluvolu, K.C., and Ang, W.T. (2011). 
    "Estimation of Physiological Tremor from Accelerometers for Real-Time Applications".
    Sensors, 11(3), 3020-3036.
    https://doi.org/10.3390/s110303020

    Parameters:
    ----------
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of harmonics from bandwidth to take in truncation.
        Bandwith endpoints are included, so n must be greater than 1.
    mu : float
        Learning rate for LMS adaptation. Positive value.
    bandwidth : tuple[float, float]
        Frequency range for bandlimited basis functions in Hz. Ex: (4, 12) Hz.
    """

    def __init__(self,
                 fs: float,
                 n: int,
                 mu: float,
                 bandwidth: tuple[float, float]):
        self.fs = fs
        self.n = n
        self.mu = mu
        self.dt = 1 / self.fs
        self.w = np.zeros(2 * n + 1)

        f_min, f_max = bandwidth
        self.ws = 2 * np.pi * (f_min + np.arange(n) *
                               (f_max - f_min) / (n - 1))

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

        for k, y in enumerate(signal):
            x = np.concatenate([
                np.sin(self.ws * k * self.dt),
                np.cos(self.ws * k * self.dt),
                [1.0]
            ])
            y_hat = self.w.T @ x
            error = y - y_hat
            self.w += 2 * self.mu * error * x

            voluntary_estimates[k] = self.w[-1]
            tremor_estimates[k] = y_hat - self.w[-1]

        return tremor_estimates, voluntary_estimates
