from __future__ import annotations
import numpy as np
from ..methodclass import Method, RunOutput


class AMOLC(Method):
    """
    Adaptive Multiple Oscillators Linear Combiner (AMOLC)
    for adaptive tremor and voluntary motion estimation.
    Refines an estimate from the 
    Bandlimited Multiple Fourier Linear Combiner with Kalman Filter (BMFLC-KF)
    by applying a Hopf-oscillator-based nonlinear correction.

    Based on the algorithm from:
    Xiao, F., Gao, Y., Wang, S., Zhao, J.,
    "Prediction of pathological tremor using adaptive multiple oscillators linear combiner",
    Biomedical Signal Processing and Control,
    Volume 27,
    2016,
    Pages 77-86,
    ISSN 1746-8094,
    https://doi.org/10.1016/j.bspc.2016.01.006.

    Parameters:
    ----------
    fs : float
        Sampling frequency in Hz.
    n : int
        Number of harmonics from bandwidth to take in truncation.
        Bandwith endpoints are included, so n must be greater than 1.
    bandwidth : tuple[float, float]
        Frequency range for bandlimited basis functions in Hz. Ex: (4, 12) Hz.
    forgetting_factor : float
        Forgetting factor for RLS. Typically close to 1 (e.g., 0.99).
    p0 : float
        Initial scale of the state error covariance matrix P = p0 * I.
        Typically a small positive value (e.g., 0.01).
    q0 : float
        Scale of the process noise covariance matrix Q = q0 * I.
        Typically a small positive value (e.g., 1e-6).
    r0: float
        Scale of the measurement noise covariance R = r0.
        Typically a small positive value (e.g., 1e-2).
    m: int
        Number of Hopf oscillators to use.
    mu: float
        Parameter of the Hopf oscillator. Typically 1.
    epsilon: float
        Hopf oscillators' learning rate.
        Determines the speed of phase synchronization
        with respect to the oscillator's input signal.
    eta: float
        Hopf oscillators' amplitude learning rate.
    """

    def __init__(self,
                 fs: float,
                 n: int,
                 bandwidth: tuple[float, float],
                 p0: float,
                 q0: float,
                 r0: float,
                 m: int,
                 mu: float,
                 epsilon: float,
                 eta: float):
        self.fs = fs
        self.n = n
        self.w = np.zeros(2 * self.n + 1)
        self.P = p0 * np.eye(2 * self.n + 1)
        self.Q = q0 * np.eye(2 * self.n + 1)
        self.R = r0
        self.m = m
        self.mu = mu
        self.epsilon = epsilon
        self.eta = eta

        self.dt = 1 / self.fs
        f_min, f_max = bandwidth
        self.ws = 2 * np.pi * (f_min + np.arange(n) *
                               (f_max - f_min) / (n - 1))

    def run(self, signal: np.ndarray) -> RunOutput:
        return RunOutput(*self._estimate_components(signal))

    def _estimate_components(self, signal: np.ndarray) -> tuple[np.ndarray,
                                                                np.ndarray,
                                                                np.ndarray]:
        # Initialize outputs
        tremor_estimates = np.zeros_like(signal)
        voluntary_estimates = np.zeros_like(signal)
        motion_estimates = np.zeros_like(signal)

        # Initialize Kalman filter variables
        I = np.eye(2 * self.n + 1)
        F = I

        # Initialize Hopf oscillators
        r = np.sqrt(self.mu) * np.ones(self.m)  # Amplitudes
        w = 2 * np.pi * np.arange(1, self.m + 1)  # Natural frequencies
        phi = np.zeros(self.m)  # Phases
        alpha = np.zeros(self.m)  # Amplitude correction terms
        theta = alpha * r * np.cos(phi)  # Oscillator outputs

        for k, z in enumerate(signal):
            # BMFLC-KF update
            x = np.concatenate([
                np.sin(self.ws * k * self.dt),
                np.cos(self.ws * k * self.dt),
                [1.0]
            ])
            w_pred = F @ self.w
            P_pred = F @ self.P @ F.T + self.Q
            S = x.T @ P_pred @ x + self.R
            y = z - x.T @ w_pred
            K = P_pred @ x / S
            self.P = (1 - self.R / S) * (I - K @ x.T) @ P_pred
            self.w = w_pred + K * y

            # Hopf oscillator-based nonlinear correction
            f = x.T @ self.w - sum(theta)  # Input to Hopf oscillators
            r += self.dt * (self.mu * r - r ** 3 +
                            self.epsilon * f * np.cos(phi))
            phi += self.dt * (w - self.epsilon * f * np.sin(phi) / r)
            w += - self.dt * self.epsilon * f * np.sin(phi)
            alpha += self.dt * self.eta * f * r * np.cos(phi)
            theta = alpha * r * np.cos(phi)

            # Store estimates
            tremor_estimates[k] = sum(theta)
            voluntary_estimates[k] = z - sum(theta)
            motion_estimates[k] = z

        return tremor_estimates, voluntary_estimates, motion_estimates
