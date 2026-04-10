from __future__ import annotations

from ..methodclass import Method, RunOutput

import numpy as np
from scipy.signal import (TransferFunction, cont2discrete,
                          sosfilt, sosfilt_zi, zpk2sos, tf2zpk)

"""
High-pass filter implementing equation (1) from
"Robust Controller for Tremor Suppression at 
Musculoskeletal Level in Human Wrist"
(Taheri et al., 2014).
"""


class HighPassFilter(Method):
    """
    High-pass filter implementing equation (1) from the 2014 Taheri et al..

    The filter is a 4th-order IIR filter with the transfer function:
    H(s) = (β₄s⁴ + β₃s³ + β₂s² + β₁s + β₀) / (α₄s⁴ + α₃s³ + α₂s² + α₁s + α₀)

    This filter is designed to separate tremor motion (3-12 Hz)
    from voluntary motion (0-2 Hz),
    with a gain of -10 dB for signals below 2 Hz.
    """

    def __init__(self,
                 fs: float):
        """
        Initialize the high-pass filter.

        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz (default: 1000 Hz)
        """
        self.fs = fs

        # Coefficients from equation (1) in the paper
        # Transfer function: H(s) = numerator / denominator
        # Coefficients in descending order of powers of s

        # Numerator coefficients: β₄s⁴ + β₃s³ + β₂s² + β₁s + β₀
        self.beta4 = 1.028
        self.beta3 = 21.23
        self.beta2 = 313.9
        self.beta1 = 2656.0
        self.beta0 = 8246.0
        numerator_continuous = np.array(
            [self.beta4, self.beta3, self.beta2, self.beta1, self.beta0],
            dtype=np.float64
        )

        # Denominator coefficients: α₄s⁴ + α₃s³ + α₂s² + α₁s + α₀
        self.alpha4 = 1.0
        self.alpha3 = 176.2
        self.alpha2 = 2227.0
        self.alpha1 = 1.512e5
        self.alpha0 = 8.411e5
        denominator_continuous = np.array(
            [self.alpha4, self.alpha3, self.alpha2, self.alpha1, self.alpha0],
            dtype=np.float64
        )

        # Create continuous-time transfer function
        sys_continuous = TransferFunction(numerator_continuous,
                                          denominator_continuous)

        # Discretize using bilinear (Tustin) transform
        # This is appropriate for control systems and
        # provides good frequency response matching
        sys_discrete = cont2discrete(
            (sys_continuous.num, sys_continuous.den),
            1.0 / fs,
            method='bilinear'
        )

        # Extract discrete-time numerator and denominator
        num_discrete = sys_discrete[0].flatten()
        den_discrete = sys_discrete[1]

        # Convert to zero-pole-gain form for better numerical stability
        z, p, k = tf2zpk(num_discrete, den_discrete)

        # Convert to second-order sections (cascade of biquads)
        # This improves numerical stability for higher-order filters
        self.sos = zpk2sos(z, p, k)

        # Initialize filter state for each second-order section
        self.zi = sosfilt_zi(self.sos)

    def run(self, signal: np.ndarray) -> RunOutput:
        """
        Apply the high-pass filter to the input signal.

        Parameters:
        -----------
        signal : np.ndarray
            Input signal array to be filtered.

        Returns:
        --------
        RunOutput
            Contains tremor_estimates (filtered output) and 
            voluntary_estimates (original signal).
        """
        tremor_estimates = self._estimate_tremor(signal)
        voluntary_estimates = signal - tremor_estimates
        motion_estimates = voluntary_estimates + tremor_estimates
        return RunOutput(tremor_estimates,
                         voluntary_estimates,
                         motion_estimates)

    def _estimate_tremor(self, signal: np.ndarray) -> np.ndarray:
        tremor_estimates = []
        self.zi *= signal[0]  # Initialize filter state with the first sample

        for sample in signal:
            # Apply cascaded second-order sections filter
            # sosfilt modifies zi in-place to maintain state
            output_array, self.zi = sosfilt(self.sos, [sample], zi=self.zi)
            output = float(output_array[0])
            tremor_estimates.append(output)

        return np.array(tremor_estimates)
