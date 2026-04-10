from __future__ import annotations
import numpy as np
from ..methodclass import Method, RunOutput
from scipy.signal import ellip, butter, sosfilt_zi, sosfilt


class WFLC(Method):
    """
    Weighted Fourier Linear Combiner (WFLC) for adaptive tremor estimation.

    Based on the algorithm from:
    Rivière, C.N. and Thakor, N.V.
    "Adaptive human-machine interface for persons with tremor"
    Proceedings of 17th International Conference of the Engineering in Medicine and Biology Society,
    Montreal, QC, Canada, 1995, pp. 1193-1194 vol.2,
    doi: 10.1109/IEMBS.1995.579637.

    The WFLC extends the FLC by adaptively tracking both tremor amplitude
    and frequency using Fourier basis functions with LMS weight updates.

    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    f0 : float
        Fundamental frequency in Hz
    n : int
        Number of harmonics to include in truncation
    mu : float
        Learning rate for LMS adaptation. Positive value.
    mu_0 : float
        Frequency adaptation gain (must be << mu for stability)
    mu_correction : float
        Adaptive gain for tremor cancelling correction.
    mu_bias : float
        Bias term for error correction in the weight update.
    filter_type : str
        Type of pre-filtering to apply to the input signal before WFLC:
        - 'highpass': High-pass filter with cutoff around 1.4 Hz (original paper)
        - 'bandpass': Band-pass filter around typical tremor frequencies i.e., 7-13 Hz
        - other: No filtering done
    flc_correction : bool
        Whether to apply FLC correction using a separate set of weights.
        If False, mu_correction is ignored.
    """

    def __init__(self,
                 fs: float,
                 f0: float,
                 n: int,
                 mu: float,
                 mu_0: float,
                 mu_correction: float,
                 mu_bias: float,
                 filter_type: str,
                 flc_correction: bool):
        self.fs = fs
        self.f0 = f0
        self.w0 = 2 * np.pi * self.f0
        self.n = n
        self.mu = mu
        self.mu_0 = mu_0
        self.dt = 1 / self.fs
        self.w = np.zeros(2 * self.n)

        # WFLC-specific variables
        self.mu_correction = mu_correction
        self.mu_bias = mu_bias
        self.w_correction = np.zeros(2 * self.n)
        self.w0_sum = self.w0
        self.flc_correction = flc_correction

        # More recent work use a bandpass filter mitigate voluntary motion,
        # original paper used a highpass filter with cutoff around 1.4 Hz.
        self.filter = None
        if filter_type == 'highpass':
            self.filter = butter(N=1,
                                 Wn=1.4,
                                 fs=self.fs,
                                 btype='high',
                                 output='sos')
        if filter_type == 'bandpass':
            self.filter = ellip(N=6,
                                rp=5,
                                rs=40,
                                Wn=[7, 13],
                                btype='bandpass',
                                fs=self.fs,
                                output='sos')

        # Initialize filter state for each second-order section
        self.zi = sosfilt_zi(self.filter)

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
        r = np.arange(self.n) + 1
        for k, y in enumerate(signal):
            # Filter the input signal
            if self.filter is not None:
                s, self.zi = sosfilt(self.filter, [y], zi=self.zi)
                s = s[0]
            else:
                s = y

            x = np.concatenate([
                np.sin(self.w0_sum * r * self.dt),
                np.cos(self.w0_sum * r * self.dt)
            ])
            s_hat = self.w.T @ x
            error = s - s_hat - self.mu_bias
            self.w += 2 * self.mu * error * x
            self.w0 += 2 * self.mu_0 * error * sum(
                r * (self.w[r - 1] * x[r + self.n - 1] -
                     self.w[r + self.n - 1] * x[r - 1])
            )
            self.w0_sum += self.w0

            if self.flc_correction:
                y_hat = np.dot(self.w_correction.T, x)
                error_y = y - y_hat
                self.w_correction += 2 * self.mu_correction * error_y * x
                tremor_estimates[k] = y_hat
            else:
                tremor_estimates[k] = s_hat

        return tremor_estimates
