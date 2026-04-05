from __future__ import annotations
from matplotlib.pylab import zeros_like
import numpy as np
from ..methodclass import Method, RunOutput


class BBF(Method):
    """
    Benedict-Bordner filter for voluntary motion estimation.
    The Benedict-Bordner filter is an alpha-beta filter (or g-h filter)
    with alpha and beta parameters related as to achieve optimal tracking.

    Implements the algorithm from:
    Benedict, T. and Bordner, G.,
    "Synthesis of an optimal set of radar track-while-scan smoothing equations"
    IRE Transactions on Automatic Control, vol. 7, no. 4, pp. 27-32, July 1962,
    doi: 10.1109/TAC.1962.1105477

    Parameters:
    -----------
    alpha : float
        Smoothing factor for the filter (0 < alpha < 1). The beta parameter is
        derived from alpha as beta = (1 - alpha)^2 / (4 * alpha).
    """

    def __init__(self, fs: float, alpha: float):
        """
        Initialize the Benedict-Bordner filter with the given alpha parameter.
        """
        if not (0 < alpha < 1):
            raise ValueError(
                "Alpha must be in the range (0, 1) for stability.")

        self.alpha = alpha
        self.fs = fs
        self.beta = (self.alpha ** 2) / (2 - self.alpha)
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
