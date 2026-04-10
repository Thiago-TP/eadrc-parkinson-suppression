from __future__ import annotations
import numpy as np
from ..methodclass import Method, RunOutput


class BMFLC_RLS(Method):
    """
    Bandlimited Multiple Fourier Linear Combiner with Recursive Least Squares (BMFLC-RLS)
    for adaptive tremor and voluntary motion estimation.
    Changes the weight update rule of the original BMFLC to a
    recursive least squares (RLS) approach for potentially faster convergence.

    Implements the algorithm from:
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
    bandwidth : tuple[float, float]
        Frequency range for bandlimited basis functions in Hz. Ex: (4, 12) Hz.
    forgetting_factor : float
        Forgetting factor for RLS. Typically close to 1 (e.g., 0.99).
    p0 : float
        Initial scale of the correlation matrix P = p0 * I.
        Typically a small positive value (e.g., 0.01).
    """

    def __init__(self,
                 fs: float,
                 n: int,
                 bandwidth: tuple[float, float],
                 forgetting_factor: float,
                 p0: float):
        self.fs = fs
        self.n = n
        self.forgetting_factor = forgetting_factor
        self.P = p0 * np.eye(2 * n + 1)
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

            # LMS update replaced with RLS update
            K = self.P @ x / (self.forgetting_factor + x.T @ self.P @ x)
            self.w += K * error
            # Numpy's outer product trick avoids changing the shape of x
            self.P = (self.P - np.outer(K, x.T @ self.P)) / self.forgetting_factor  # noqa: E501

            voluntary_estimates[k] = self.w[-1]
            tremor_estimates[k] = y_hat - self.w[-1]

        return tremor_estimates, voluntary_estimates
