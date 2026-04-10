from __future__ import annotations

from ..methodclass import Method, RunOutput

import numpy as np
from scipy.signal import (TransferFunction, cont2discrete,
                          sosfilt, sosfilt_zi, zpk2sos, tf2zpk)


class ABPF(Method):
    """
    Adaptive Bandpass Filter (ABPF) for tremor estimation.

    Implements the algorithm from:
    Popović, L.Z. and Šekara, T.B. and Popović, M.B.,
    "Adaptive band-pass filter (ABPF) for tremor extraction from inertial sensor data",
    Methods and Programs in Biomedicine, Volume 99, Issue 3, 2010, Pages 298-305, ISSN 0169-2607,
    https://doi.org/10.1016/j.cmpb.2010.03.018.

    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    f_center : float
        Initial center frequency for the bandpass filter in Hz
    delta_f : float
        Frequency step for the damping block in Hz
    slack : float
        Maximum allowed deviation of the center frequency from the initial center frequency in Hz
    """

    def __init__(self,
                 fs: float,
                 f_center: float,
                 delta_f: float,
                 slack: float):
        """Initialize the ABPF algorithm."""
        self.fs = fs
        self.f_center = f_center
        self.delta_f = delta_f
        self.slack = slack

        self.dt = 1 / self.fs
        self.f_old = f_center
        self.f_in = f_center

        # Coefficients from equation (1) in the paper
        # Transfer function: H(s) = numerator / denominator
        # Coefficients in descending order of powers of s

        # Numerator coefficients: βw_as + 0
        self.beta = np.sqrt(2)
        self.w_a = 2 * np.pi * self.f_center
        self.sos = None
        self._update_bandpass_filter()

        # Initialize filter state for each second-order section
        self.zi = sosfilt_zi(self.sos)

    def _update_bandpass_filter(self) -> None:
        """Update the bandpass filter based on current center frequency."""
        numerator_continuous = np.array([
            self.beta * self.w_a, 0
        ])
        denominator_continuous = np.array([
            1, self.beta * self.w_a, self.w_a ** 2
        ])
        sys_continuous = TransferFunction(
            numerator_continuous,
            denominator_continuous
        )
        sys_discrete = cont2discrete(
            (sys_continuous.num, sys_continuous.den),
            self.dt,
            method='bilinear'
        )
        num_discrete = sys_discrete[0].flatten()
        den_discrete = sys_discrete[1]
        z, p, k = tf2zpk(num_discrete, den_discrete)
        self.sos = zpk2sos(z, p, k)

    def _damping_block(self) -> None:
        if self.f_in >= self.f_old + self.delta_f:
            self.f_out = self.f_old + self.delta_f

            self.f_old = self.f_out
            return

        if (self.f_in >= self.f_old + 2 * self.delta_f and
                self.f_old <= self.f_center - 2 * self.delta_f):
            self.f_out = self.f_old + 2*self.delta_f

            self.f_old = self.f_out
            return

        if self.f_in <= self.f_old - self.delta_f:
            self.f_out = self.f_old - self.delta_f

            self.f_old = self.f_out
            return

        if (self.f_in <= self.f_old - 2 * self.delta_f and
                self.f_old >= self.f_center + 2 * self.delta_f):
            self.f_out = self.f_old - 2*self.delta_f

            self.f_old = self.f_out
            return

    def _frequency_limiting_block(self) -> None:

        if self.f_out > self.f_center + self.slack:
            self.f_center = self.f_center + self.slack

        if self.f_out < self.f_center - self.slack:
            self.f_center = self.f_center - self.slack

        self.w_a = 2 * np.pi * self.f_center

    def run(self, signal: np.ndarray) -> RunOutput:
        """Run the ABPF algorithm on the input signal."""
        # Placeholder implementation
        mean = np.mean(signal)
        tremor_estimates = self._estimate_tremor(signal - mean)
        voluntary_estimates = signal - tremor_estimates + mean
        motion_estimates = voluntary_estimates + tremor_estimates
        return RunOutput(
            tremor_estimates=tremor_estimates,
            voluntary_estimates=voluntary_estimates,
            motion_estimates=motion_estimates
        )

    def _estimate_tremor(self, signal: np.ndarray) -> np.ndarray:
        tremor_estimates = []
        h = 0.0  # time since last crossing

        for k, s in enumerate(signal):

            h += self.dt

            # Check if crossing occurred (sign change in signal)
            if s * signal[k-1] < 0:
                # Update initial frequency estimate based on h
                self.f_in = 1 / (2 * h)

                # Update center frequency based on f_in, f_delta, etc.
                self._damping_block()
                self._frequency_limiting_block()
                self._update_bandpass_filter()

                # Reset timer since last crossing
                h = 0.0

            # Apply the bandpass filter to the signal
            tremor_estimate, self.zi = sosfilt(self.sos, [s], zi=self.zi)
            tremor_estimates.append(tremor_estimate[0])

        return np.array(tremor_estimates)
