"""
Method parameter initialization and management.

This module centralizes default parameter definitions
for all tremor estimation methods.
"""
from utils.constants import FS
from typing import Union
import numpy as np
from scipy.linalg import block_diag

ParameterDict = Union[
    dict[str, int | float],
    dict[str, int | float | str],
    dict[str, int | float | tuple[int, int]],
    dict[str, int | float | np.ndarray]
]


def get_method_parameters(method_name: str) -> ParameterDict:
    """
    Get initialization parameters for a tremor estimation method.

    Parameters:
    -----------
    method_name : str
        Tremor estimation method class name/nickname (e.g., WFLC)

    Returns:
    --------
    ParameterDict
        Dictionary of parameters for initializing the method
    """

    # Method parameter definitions
    parameters = {
        'LowPassFilter': {
            "fs": FS,
            "cutoff_freq": 1.5,
            "order": 1
        },
        'HighPassFilter': {
            "fs": FS
        },
        'FLC': {
            "fs": FS,
            "f0": 0.5,  # changes between test cases
            "n": 2,
            "mu": 0.005
        },
        'WFLC': {
            "fs": FS,
            "f0": 14,
            "n": 1,
            "mu": 5e-3,
            "mu_0": 5e-4,
            "mu_correction": 0.01,
            "mu_bias": 0,
            "filter_type": 'bandpass',
            "flc_correction": True,
        },
        'BMFLC': {
            "fs": FS,
            "n": 2,
            "mu": 0.01,
            "bandwidth": (12, 14),
        },
        'EBMFLC': {
            "fs": FS,
            "n": 200,
            "mu": 1e-3,
            "total_bandwidth": (0, 14),
            "voluntary_bandwidth": (0, 4),
            "tremor_bandwidth": (6, 14),
            "window_time": 3.0,
            "minimum_impact": 0.01,
        },
        'ABPF': {
            "fs": FS,
            "f_center": 8.0,
            "delta_f": 0.06,
            "slack": 1.5,
        },
        'BBF': {
            "fs": FS,
            "alpha": 0.018,
        },
        'CDF': {
            "fs": FS,
            "theta": 0.99,
        },
        'KF': {
            "fs": FS,
            "cov_process": 1e-1,
            "cov_measurement": 1e-6,
            "cov_position": 1e-7,
            "cov_velocity": 1e-7,
        },
        'WFLCKF': {
            "fs": FS,
            "theta": 0.99,
            "f0": 14,
            "n": 1,
            "mu": 5e-3,
            "mu_0": 5e-4,
            "mu_bias": 1e-7,
            "cov_process": 1e0,
            "cov_measurement": 1e-7,
        },
        'BMFLC_RLS': {
            "fs": FS,
            "n": 2,
            "bandwidth": (12, 14),
            "forgetting_factor": 0.95,
            "p0": 1e-2
        },
        'BMFLC_KF': {
            "fs": FS,
            "n": 2,
            "bandwidth": (12, 14),
            "p0": 1e-1,
            "q0": 1e-1,
            "r0": 3,
        },
        'AR_LMS': {
            "fs": FS,
            "m": 5,
            "mu": 0.02,
        },
        'AR_KF': {
            "fs": FS,
            "m": 2,
            "p0": 1e-3,
            "q0": 1e-2,
            "r0": 1e-3,
        },
        'AS_BMFLC': {
            "fs": FS,
            "n": 2,
            "f0": 13,
            "mu_wflc": 5e-2,
            "mu_0_wflc": 5e-2,
            "mu_bmflc": 1e-2,
            "bandwidth": (12, 14),
            "window_length": 0.5,
        },
        'ZPAFKF': {
            "fs": FS,
            "cov_process": 1e-2,
            "cov_measurement": 1e-2,
            "cov_position": 1e-5,
            "cov_velocity": 1e-5,
            "cov_acceleration": 1e-5,
            "m": 3,
            "delta_R": 5e-1,
        },
        'AMOLC': {
            "fs": FS,
            "n": 2,
            "bandwidth": (12, 14),
            "p0": 1e-1,
            "q0": 1e-1,
            "r0": 3,
            "m": 13,
            "mu": 1.0,
            "epsilon": 1.0,
            "eta": 3.0,
        },
        'EHWFLC_KF': {
            "fs": FS,
            "f0": 12,
            "n": 3,
            "mu": 1e-1 * np.eye(2 * 3),
            "mu_0": 1e-3 * np.ones(3),
            "P": 1e-5 * block_diag(*[np.eye(2) for _ in range(3)]),
            "Q": 1e-5 * np.eye(2 * 3),
            "R": 1e-5,
        },
    }

    if method_name not in parameters:
        raise ValueError(f"Unknown method name '{method_name}'. "
                         f"Available methods: {list(parameters.keys())}")
    return parameters[method_name]
