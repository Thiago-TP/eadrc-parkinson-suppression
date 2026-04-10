from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ..methodclass import Method, RunOutput


class AR_KF(Method):
    """
    Autoregressive model with Least Mean Squares coefficients update for tremor estimation (AR-KF).
    Raw signal is bandpass filtered to isolate tremor frequencies,
    then an AR model is fitted to the signal with coefficients updated
    using the LMS algorithm. The AR model output is taken as the tremor
    estimate, and remaining motion in the input is considered voluntary.
    The bandpass filter is a fifth order butterworth.

    Based on the algorithm from:

    Tatinati, S. and Veluvolu, K.C. and Hong, S.M. and Latt, W. T. and Ang, W. T.,
    "Physiological Tremor Estimation With Autoregressive (AR) Model and Kalman Filter for Robotics Applications,"
    in IEEE Sensors Journal, vol. 13, no. 12, pp. 4977-4985, Dec. 2013,
    doi: 10.1109/JSEN.2013.2271737

    Parameters:
    -----------
    fs : float
        Sampling frequency of the input signal (Hz)
    m : int
        Order of the autoregressive model
    p0 : float
        Initial state error covariance for the Kalman filter
    q0 : float
        Process error covariance for the Kalman filter
    r0 : float
        Observation error covariance for the Kalman filter
    """

    def __init__(self,
                 fs: float,
                 m: int,
                 p0: float,
                 q0: float,
                 r0: float):
        # Filter parameters: sampling frequency
        self.fs = fs
        self.filter = butter(
            N=1,
            Wn=[5, 14],
            btype='bandpass',
            fs=fs,
            output='sos'
        )

        # AR parameters: order
        self.m = m

        # KF parameters: state, process and observation error covariances
        self.P = p0 * np.eye(self.m)
        self.Q = q0 * np.eye(self.m)
        self.R = r0

        # AR coefficients and inputs
        self.w = np.zeros(self.m)
        self.x = np.zeros(self.m)

    def run(self, signal: np.ndarray) -> RunOutput:
        tremor_estimates = self._estimate_tremor(signal)
        voluntary_estimates = signal - tremor_estimates
        motion_estimates = voluntary_estimates + tremor_estimates
        return RunOutput(
            tremor_estimates=tremor_estimates,
            voluntary_estimates=voluntary_estimates,
            motion_estimates=motion_estimates
        )

    def _estimate_tremor(self, signal: np.ndarray) -> np.ndarray:

        tremor_estimates = np.zeros_like(signal)

        # Initialize filter state
        zi = sosfilt_zi(self.filter) * signal[0]
        I = np.eye(self.m)

        for n, s_n in enumerate(signal):
            # Bandpass filter the sample to isolate tremor frequencies
            x_n, zi = sosfilt(self.filter, [s_n], zi=zi)
            x_n = x_n[0]  # Extract scalar from array

            # Update estimates with AR model prediction
            x_n_hat = self.w.T @ self.x
            tremor_estimates[n] = x_n_hat
            # tremor_estimates[n] = x_n

            # KF update
            error = x_n - x_n_hat
            K = self.P @ self.x / (self.x.T @ self.P @ self.x + self.R)
            self.P = (I - K @ self.x.T) @ self.P + self.Q
            self.w += K * error

            # AR inputs update
            self.x = np.roll(self.x, 1)
            self.x[0] = x_n

        return tremor_estimates
