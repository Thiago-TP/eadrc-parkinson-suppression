from __future__ import annotations
import numpy as np
from ..methodclass import Method, RunOutput


class BMFLC_KF(Method):
    """
    Bandlimited Multiple Fourier Linear Combiner with Kalman Filter (BMFLC-KF)
    for adaptive tremor and voluntary motion estimation.
    Changes the weight update rule of the original BMFLC to a
    Kalman filter (KF) approach for potentially faster convergence.

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
        Initial scale of the state error covariance matrix P = p0 * I.
        Typically a small positive value (e.g., 0.01).
    q0 : float
        Scale of the process noise covariance matrix Q = q0 * I.
        Typically a small positive value (e.g., 1e-6).
    r0: float
        Scale of the measurement noise covariance R = r0.
        Typically a small positive value (e.g., 1e-2).
    """

    def __init__(self,
                 fs: float,
                 n: int,
                 bandwidth: tuple[float, float],
                 p0: float,
                 q0: float,
                 r0: float):
        self.fs = fs
        self.n = n
        self.w = np.zeros(2 * self.n + 1)
        self.P = p0 * np.eye(2 * self.n + 1)
        self.Q = q0 * np.eye(2 * self.n + 1)
        self.R = r0
        self.dt = 1 / self.fs

        f_min, f_max = bandwidth
        self.ws = 2 * np.pi * (f_min + np.arange(n) *
                               (f_max - f_min) / (n - 1))

    def run(self, signal: np.ndarray) -> RunOutput:
        return RunOutput(*self._estimate_components(signal))

    def _estimate_components(self, signal: np.ndarray) -> tuple[np.ndarray,
                                                                np.ndarray,
                                                                np.ndarray]:
        tremor_estimates = np.zeros_like(signal)
        voluntary_estimates = np.zeros_like(signal)
        motion_estimates = np.zeros_like(signal)
        I = np.eye(2 * self.n + 1)
        F = I

        for k, z in enumerate(signal):
            x = np.concatenate([
                np.sin(self.ws * k * self.dt),
                np.cos(self.ws * k * self.dt),
                [1.0]
            ])

            # LMS update replaced with KF update
            w_pred = F @ self.w  # Predicted state
            P_pred = F @ self.P @ F.T + self.Q  # Predicted error covariance
            S = x.T @ P_pred @ x + self.R  # Innovation covariance
            y = z - x.T @ w_pred  # Innovation
            K = P_pred @ x / S  # Kalman gain
            self.P = (1 - self.R / S) * (I - K @ x.T) @ P_pred
            self.w = w_pred + K * y

            voluntary_estimates[k] = self.w[-1]
            tremor_estimates[k] = x.T @ self.w - self.w[-1]
            motion_estimates[k] = x.T @ self.w

        return tremor_estimates, voluntary_estimates, motion_estimates
