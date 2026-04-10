from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag
from ..methodclass import Method, RunOutput
from scipy.signal import butter, sosfilt, sosfilt_zi


class EHWFLC_KF(Method):
    """
    Enhanced High-Order WFLC-based Kalman Filter (EHWFLC-KF)
    for tremor estimation.

    Raw input signal is bandpassed on tremor frequency range,
    then has its frequency estimated by a high-order WFLC,
    and then has its amplitude estimated by a Kalman Filter (KF).
    The order of WFLC is increased by using a learning rate for each harmonic,
    instead of the original single rate for all harmonics.
    A 2nd-order Butterworth zero-phase band-pass filter
    with cutoff frequencies of 1 and 30 Hz is used to prepare the raw signal.

    Based on the algorithm from:
    Y. Zhou and M. E. Jenkins and M. D. Naish and A. L. Trejos
    "Characterization of Parkinsonian Hand Tremor and Validation of a High-Order Tremor Estimator,"
    in IEEE Transactions on Neural Systems and Rehabilitation Engineering,
    vol. 26, no. 9, pp. 1823-1834, Sept. 2018,
    doi: 10.1109/TNSRE.2018.2859793.

    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    f0 : float
        Fundamental frequency in Hz
    n : int
        Number of harmonics to include in truncation
    mu : np.ndarray
        Learning rates for LMS adaptation of each harmonic (2n by 2n).
    mu_0 : np.ndarray
        Frequency adaptation gains for each harmonic.
    P : np.ndarray
        Initial state covariance matrix for the KF (2n by 2n).
    Q : np.ndarray
        Process noise covariance matrix for the KF (2n by 2n).
    R : float
        Measurement noise covariance matrix for the KF.
    """

    def __init__(self,
                 fs: float,
                 f0: float,
                 n: int,
                 mu: np.ndarray,
                 mu_0: np.ndarray,
                 P: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray):
        self.fs = fs
        self.f0 = f0
        self.n = n
        self.mu = mu.copy()
        self.mu_0 = mu_0.copy()
        self.w = np.zeros(2 * self.n)
        self.w0 = 2 * np.pi * self.f0 * np.ones(self.n)
        self.w0_sum = 2 * np.pi * self.f0 * np.ones(self.n)
        self.filter = butter(  # not sure how they got a zero-phase...
            N=2,
            Wn=[1, 30],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        self.P = P.copy()
        self.Q = Q.copy()
        self.R = R

    def run(self, signal: np.ndarray) -> RunOutput:
        tremor_estimates = self._estimate_tremor(signal)
        voluntary_estimates = signal - tremor_estimates
        motion_estimates = tremor_estimates + voluntary_estimates
        return RunOutput(
            tremor_estimates=tremor_estimates,
            voluntary_estimates=voluntary_estimates,
            motion_estimates=motion_estimates
        )

    def _estimate_tremor(self, signal: np.ndarray) -> np.ndarray:
        tremor_estimates = np.zeros_like(signal)
        r = np.arange(self.n) + 1
        zi = sosfilt_zi(self.filter) * signal[0]
        dt = 1 / self.fs

        self.x = np.ones((2 * self.n, 1))  # KF state
        I = np.eye(2 * self.n)
        H = np.zeros((1, 2 * self.n))
        for i in range(2 * self.n):
            if i % 2 == 0:
                H[0, i] = 1.0

        for k, y in enumerate(signal):
            # Filter the input signal
            s, zi = sosfilt(self.filter, [y], zi=zi)
            s = s[0]

            # HWFLC estimate of frequency
            x = np.concatenate([
                [np.sin(w * dt) for w in self.w0_sum],
                [np.cos(w * dt) for w in self.w0_sum]
            ])
            s_hat = x.T @ self.w
            error = s - s_hat
            self.w += 2 * self.mu @ x * error
            for i in range(self.n):
                self.w0[i] += 2 * self.mu_0[i] * error * (
                    self.w[i] * x[i + self.n] -
                    self.w[i + self.n] * x[i]
                )
            print(self.w0)
            self.w0_sum += self.w0

            # KF update of amplitude
            F = block_diag(*[
                [
                    [np.cos(w * dt), np.sin(w * dt) / (w * dt)],
                    [-np.sin(w * dt) / (w * dt), np.cos(w * dt)],
                ]
                for w in self.w0_sum
            ])
            x_pred = F @ self.x
            P_pred = F @ self.P @ F.T + self.Q
            y = s - (H @ x_pred).item()
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T / S
            self.x = x_pred + K * y
            self.P = (I - K @ H) @ P_pred
            z = H @ self.x

            tremor_estimates[k] = z.item()

        return tremor_estimates
