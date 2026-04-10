from __future__ import annotations
from matplotlib.pylab import zeros_like
import numpy as np
from ..methodclass import Method, RunOutput


class CDF(Method):
    """
    Critically Damped Filter for voluntary motion estimation.
    The CDF is an alpha-beta filter (or g-h filter)
    with alpha and beta parameters related as to minimize the least squares
    fitting line of previous measurements.

    Implements the algorithm from (eqs. 5-8, 10):
    Gallego, J. A. and Rocon, E. and Roa, J. O. and Moreno, J. C. and Pons, J. L. (2010).
    "Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data".
    Sensors, 10(3), 2129-2149.
    https://doi.org/10.3390/s100302129

    Parameters:
    -----------
    fs: float
        Sampling frequency of the input signal.
    theta : float
        Damping factor for the filter (0 < theta < 1).
    """

    def __init__(self, fs: float, theta: float):
        """
        Initialize the Critically Damped Filter with the given theta parameter.
        """
        if not (0 < theta < 1):
            raise ValueError(
                "Theta must be in the range (0, 1) for stability.")
        self.theta = theta
        self.fs = fs
        self.alpha = 1 - (self.theta ** 2)
        self.beta = (1 - self.theta) ** 2
        self.dt = 1 / self.fs

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
        """Standard alpha-beta filter loop."""
        voluntary_estimates = zeros_like(signal)
        x = 0.0
        v = 0.0

        for k, s in enumerate(signal):

            r = s - x
            x += self.alpha * r
            v += (self.beta / self.dt) * r
            x += (v * self.dt)

            voluntary_estimates[k] = x

        return voluntary_estimates
