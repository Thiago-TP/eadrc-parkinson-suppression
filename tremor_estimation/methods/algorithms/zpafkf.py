from __future__ import annotations
import numpy as np
from scipy.signal import ellip, sosfilt, sosfilt_zi
from ..methodclass import Method, RunOutput


class ZPAFKF(Method):
    """
    Zero-phase filter plus adaptive fuzzy Kalman Filter (ZPAFKF)
    for tremor estimation.
    This method works as follows:
    1. Highpass elliptic filter (HPEF) is applied to the raw input signal
    2. Lowpass elliptic filter with equal but opposite fase of HPEF is applied to its output
    3. A standard Kalman Filter (KF) is applied to the lowpass-filtered signal, returning an estimate of the tremor
    4. Measurement noise covariance matriz of KF is updated using fuzzy logic

    Based on the algorithm from:
    Sang, H. and Yang, C. and Liu, F. and Yun, J. and Jin, G. and Chen, F.
    (2016)
    "A zero phase adaptive fuzzy Kalman filter for physiological tremor suppression in robotically assisted minimally invasive surgery".
    Int J Med Robotics Comput Assist Surg, 12: 658-669.
    doi: 10.1002/rcs.1741.

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
    cov_acceleration: float
        Acceleration state covariance.
    m: int
        Amount of past Kalman filter innovations to consider during
        fuzzy update of measurement noise covariance.
    delta_R: float
        Step size for fuzzy update of measurement noise covariance.
    """

    def __init__(self,
                 fs: float,
                 cov_process: float,
                 cov_measurement: float,
                 cov_position: float,
                 cov_velocity: float,
                 cov_acceleration: float,
                 m: int,
                 delta_R: float):
        self.fs = fs
        self.m = m
        self.delta_R = delta_R
        self.dt = 1 / fs

        # Elliptic filters
        self.hpef = ellip(
            N=2,
            rp=1,
            rs=60,
            Wn=1.595,
            fs=self.fs,
            btype='highpass',
            output='sos'
        )
        self.lpef = ellip(
            N=2,
            rp=1,
            rs=57.699,
            Wn=57.325,
            fs=self.fs,
            btype='lowpass',
            output='sos'
        )

        # State transition matrix
        self.F = np.array([
            [1, self.dt, self.dt ** 2 / 2],
            [0, 1, self.dt],
            [0, 0, 1]
        ])

        # Measurement matrix
        self.H = np.array([[1, 0, 0]])

        # Process noise covariance
        self.Q = cov_process * np.eye(3)

        # Measurement noise covariance
        self.R = np.array([[cov_measurement]])

        # State: voluntary position and velocity
        self.x = np.zeros((3, 1))

        # Initial estimate covariance
        self.P = np.array([
            [cov_position, 0, 0],
            [0, cov_velocity, 0],
            [0, 0, cov_acceleration]
        ])

    def run(self, signal: np.ndarray) -> RunOutput:
        tremor_estimates = self._estimate_tremor(signal)
        voluntary_estimates = signal - tremor_estimates
        motion_estimates = voluntary_estimates + tremor_estimates
        return RunOutput(
            voluntary_estimates=voluntary_estimates,
            tremor_estimates=tremor_estimates,
            motion_estimates=motion_estimates
        )

    def _estimate_tremor(self, signal: np.ndarray) -> np.ndarray:

        tremor_estimates = np.zeros_like(signal)
        I = np.eye(3)
        zi = sosfilt_zi(self.hpef) * signal[0]
        r = np.zeros(self.m)  # innovations buffer

        for k, s in enumerate(signal):
            # Zero-phase filtering
            z, zi = sosfilt(self.hpef, [s], zi=zi)
            z, zi = sosfilt(self.lpef, z, zi=zi)
            z = z[0]

            # Kalman Filter predictions and updates
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

            # Adaptive fuzzy update of measurement noise covariance R
            r = np.roll(r, -1)
            r[-1] = y
            C = sum(r ** 2) / self.m
            DoM = S - C
            if DoM > 0:
                self.R += self.delta_R  # Increase measurement noise covariance
            elif DoM < 0:
                self.R -= self.delta_R  # Decrease measurement noise covariance

            # Tremor estimate
            tremor_estimates[k] = z.item()

        return tremor_estimates
