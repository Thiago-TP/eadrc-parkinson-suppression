from __future__ import annotations
import numpy as np
from ..methodclass import Method, RunOutput

from scipy.signal import butter, sosfilt, sosfilt_zi
"""
Simple low-pass filter (LPF) that baselines estimation of tremor.
Voluntary component is given as the filtered output of the original signal.
Tremor component is given as original signal minus its voluntary component.
"""


class LowPassFilter(Method):
    """
    Simple low-pass filter (LPF) that baselines estimation of tremor.
    Voluntary component is given as the filtered output of the original signal.
    Tremor component is given as original signal minus its voluntary component.
    """

    def __init__(self,
                 fs: float,
                 cutoff_freq: float,
                 order: int):
        """
        Initialize the low-pass filter.

        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz
        cutoff_freq : float
            Cutoff frequency in Hz
        order : int
            Filter order (e.g., 4 for a 4th-order Butterworth filter)
        """
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = order
        self.lowpass_filter = butter(N=self.order,
                                     Wn=self.cutoff_freq,
                                     fs=self.fs,
                                     btype='low',
                                     output='sos')
        self.zi = sosfilt_zi(self.lowpass_filter)

    def _estimate_voluntary(self, signal: np.ndarray) -> np.ndarray:
        """
        Estimate voluntary component by applying a low-pass filter to the input signal.

        Parameters:
        -----------
        signal : np.ndarray
            Input signal from which to estimate voluntary component

        Returns:
        --------
        np.ndarray
            Estimated voluntary component (low-pass filtered signal)
        """

        # Apply zero-phase filtering to avoid phase distortion
        voluntary_estimates = []
        self.zi *= signal[0]  # Initialize filter state with the first sample

        for sample in signal:
            voluntary_estimate, self.zi = sosfilt(self.lowpass_filter,
                                                  [sample],
                                                  zi=self.zi)
            voluntary_estimates.append(voluntary_estimate[0])

        return np.array(voluntary_estimates)

    def run(self, signal: np.ndarray) -> RunOutput:
        """
        Run the low-pass filter on the input signal to separate 
        voluntary and tremor components.

        Parameters:
        -----------
        signal : np.ndarray
            Input signal from which to separate voluntary and tremor components

        Returns:
        --------
        RunOutput with estimated tremor and voluntary components.
        """
        voluntary_estimates = self._estimate_voluntary(signal)
        tremor_estimates = signal - voluntary_estimates
        motion_estimates = voluntary_estimates + tremor_estimates
        return RunOutput(tremor_estimates,
                         voluntary_estimates,
                         motion_estimates)
