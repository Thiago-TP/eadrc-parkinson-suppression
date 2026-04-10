from __future__ import annotations
import numpy as np
from ..methodclass import Method, RunOutput


class KF(Method):
    """
    Kalman Filter for voluntary motion tracking.
    A standard Kalman Filter with a first-order state-space model of the signal.

    Implements the algorithm from (eqs. 11-13):
    Gallego, J. A. and Rocon, E. and Roa, J. O. and Moreno, J. C. and Pons, J. L. (2010).
    "Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data".
    Sensors, 10(3), 2129-2149.
    https://doi.org/10.3390/s100302129

    Parameters:
    -----------
    fs: float
        Sampling frequency of the input signal.
    cov_process: float
        Process noise covariance.
    cov_measurement: float
        Measurement noise covariance.
    cov_position: float
        Position state covariance.
    cov_velocity: float
        Velocity state covariance.
    """

    def __init__(self,
                 fs: float,
                 cov_process: float,
                 cov_measurement: float,
                 cov_position: float,
                 cov_velocity: float):
        self.fs = fs
        self.dt = 1 / fs

        # State transition matrix
        self.F = np.array([
            [1, self.dt],
            [0, 1]
        ])

        # Measurement matrix
        self.H = np.array([[1, 0]])

        # Process noise covariance
        self.Q = cov_process * np.array([
            [(self.dt ** 4) / 4, (self.dt ** 3) / 2],
            [(self.dt ** 3) / 2, self.dt ** 2]
        ])

        # Measurement noise covariance
        self.R = np.array([[cov_measurement]])

        # State: voluntary position and velocity
        self.x = np.zeros((2, 1))

        # Initial estimate covariance
        self.P = np.array([
            [cov_position, 0],
            [0, cov_velocity]
        ])

    def run(self, signal: np.ndarray) -> RunOutput:
        voluntary_estimates = self._estimate_voluntary(signal)
        tremor_estimates = signal - voluntary_estimates
        motion_estimates = voluntary_estimates + tremor_estimates
        return RunOutput(
            voluntary_estimates=voluntary_estimates,
            tremor_estimates=tremor_estimates,
            motion_estimates=motion_estimates
        )

    def _estimate_voluntary(self, signal: np.ndarray) -> np.ndarray:
        """Standard Kalman Filter loop."""

        voluntary_estimates = np.zeros_like(signal)
        I = np.eye(2)

        for k, z in enumerate(signal):
            # Prediction step
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q

            # Measurement update step
            y = z - (self.H @ x_pred).item()  # Measurement residual
            S = self.H @ P_pred @ self.H.T + self.R  # Residual covariance
            K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain

            # Update state estimate and covariance
            self.x = x_pred + K * y
            self.P = (I - K @ self.H) @ P_pred
            z = self.H @ self.x  # Measurement prediction

            # Voluntary position estimate
            voluntary_estimates[k] = z.item()

        return voluntary_estimates
