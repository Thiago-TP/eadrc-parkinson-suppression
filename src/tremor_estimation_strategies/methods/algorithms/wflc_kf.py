from __future__ import annotations
import numpy as np

from ..methodclass import Method, RunOutput


class WFLCKF(Method):
    """
    Weighted-frequency Fourier Linear Combiner with Kalman Filter (WFLC-KF) 
    for tremor estimation.
    Combines the adaptive frequency tracking of WFLC with a Kalman Filter
    to improve tremor amplitude estimation.

    Implements the algorithm from (eqs. 14-17, 22-23):
    Gallego, J. A. and Rocon, E. and Roa, J. O. and Moreno, J. C. and Pons, J. L. (2010).
    "Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data".
    Sensors, 10(3), 2129-2149.
    https://doi.org/10.3390/s100302129

    The method proceeds as follows (see Figure 8 in the paper):
    1. Voluntary motion in the raw signal is estimated by
        a critically damped alpha-beta (or g-h) filter (see cdf.py)
    2. Estimated voluntary motion is subtracted from the raw signal
    3. Resulting signal is fed into a WFLC to estimate the tremor frequency
    4. WFLC and CDF outputs are fed to KF to refine tremor amplitude estimate

    This method's WFLC does use a FLC for correction.
    """

    def __init__(self,
                 # CDF parameters
                 fs: float,
                 theta: float,
                 # WFLC parameters
                 f0: float,
                 n: int,
                 mu: float,
                 mu_0: float,
                 mu_bias: float,
                 # KF parameters
                 cov_process: float,
                 cov_measurement: float,):
        """Initialize WFLC-KF."""
        # CDF setup
        if not (0 < theta < 1):
            raise ValueError(
                "Theta must be in the range (0, 1) for stability.")
        self.theta = theta
        self.fs = fs
        self.alpha = 1 - (self.theta ** 2)
        self.beta = (1 - self.theta) ** 2
        self.dt = 1 / self.fs

        # WFLC setup
        self.fs = fs
        self.f0 = f0
        self.w0 = 2 * np.pi * self.f0
        self.n = n
        self.mu = mu
        self.mu_0 = mu_0
        self.w = np.zeros(2 * self.n)
        self.mu_bias = mu_bias
        self.w0_sum = self.w0

        # KF setup (see kalman_filter.py)
        self.F = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, np.cos(self.w0 * self.dt), np.sin(self.w0 * self.dt), 0]
        ])
        self.H = np.array([[0, 0, 0, 1]])
        self.Q = cov_process * np.eye(4)
        self.R = np.array([[cov_measurement]])
        self.x = np.zeros((4, 1))
        self.P = 1e-3 * np.eye(4)

    def run(self, signal: np.ndarray) -> RunOutput:
        tremor_estimates = self._estimate_tremor(signal)
        voluntary_estimates = signal - tremor_estimates
        motion_estimates = tremor_estimates
        return RunOutput(
            tremor_estimates=tremor_estimates,
            voluntary_estimates=voluntary_estimates,
            motion_estimates=motion_estimates
        )

    def _estimate_tremor(self, signal: np.ndarray) -> np.ndarray:

        tremor_estimates = np.zeros_like(signal)
        _x = 0.0  # voluntary position estimate
        _v = 0.0  # voluntary velocity estimate
        I = np.eye(4)
        r = np.arange(self.n) + 1

        for k, s in enumerate(signal):
            # CDF step: remove voluntary motion from signal
            _r = s - _x
            _x += self.alpha * _r
            _v += (self.beta / self.dt) * _r
            _x += (_v * self.dt)

            # WFLC step: estimate tremor frequency
            tr = s - _x  # CDF-based tremor estimate

            x = np.concatenate([
                np.sin(self.w0_sum * r * self.dt),
                np.cos(self.w0_sum * r * self.dt)
            ])
            tr_hat = self.w.T @ x
            error = tr - tr_hat - self.mu_bias
            self.w += 2 * self.mu * error * x
            self.w0 += 2 * self.mu_0 * error * sum(
                r * (self.w[r - 1] * x[r + self.n - 1] -
                     self.w[r + self.n - 1] * x[r - 1])
            )
            self.w0_sum += self.w0

            # KF step: tremor amplitude refinement
            self.F[3, 1] = np.cos(self.w0)
            self.F[3, 2] = np.sin(self.w0)
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q
            y = tr - (self.H @ x_pred).item()
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            self.x = x_pred + K * y
            self.P = (I - K @ self.H) @ P_pred
            z = self.H @ self.x

            # Update estimates
            tremor_estimates[k] = z.item()

        return tremor_estimates
