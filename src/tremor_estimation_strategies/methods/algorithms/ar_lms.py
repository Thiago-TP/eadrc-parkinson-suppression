from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ..methodclass import Method, RunOutput


class AR_LMS(Method):
    """
    Autoregressive model with Least Mean Squares coefficients update for tremor estimation (AR-LMS).
    Raw signal is bandpass filtered to isolate tremor frequencies,
    then an AR model is fitted to the signal with coefficients updated
    using the LMS algorithm. The AR model output is taken as the tremor
    estimate, and remaining motion in the input is considered voluntary.
    The bandpass filter is a fifth order butterworth with 2-20 Hz band.

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
    mu : float
        Step size for the LMS coefficient update (e.g., 0.01)
    """

    def __init__(self,
                 fs: float,
                 m: int,
                 mu: float):
        # Bandpass filter parameters: sampling frequency
        self.fs = fs
        self.filter = butter(
            N=5,
            Wn=[5, 14],
            btype='bandpass',
            fs=fs, output='sos')

        # AR parameters: order and LMS step size
        self.m = m
        self.mu = mu

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

        for n, s_n in enumerate(signal):
            # Bandpass filter the sample to isolate tremor frequencies
            x_n, zi = sosfilt(self.filter, [s_n], zi=zi)
            x_n = x_n[0]  # Extract scalar from array

            # Update estimates with AR model prediction
            x_n_hat = self.w.T @ self.x
            tremor_estimates[n] = x_n_hat

            # LMS update
            error = x_n - x_n_hat
            self.w += 2 * self.mu * error * self.x

            # AR inputs update
            self.x = np.roll(self.x, 1)
            self.x[0] = x_n

        return tremor_estimates
