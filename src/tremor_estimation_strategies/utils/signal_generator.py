from __future__ import annotations
import numpy as np
import os
from utils.constants import (INPUT_DIR,
                             TREMOR_MODULATION_AMPLITUDE,
                             FIXED_FREQUENCY_NPZ,
                             MODULATED_FREQUENCY_NPZ,
                             FIXED_FREQUENCY_VALUES,
                             MODULATED_FREQUENCY_VALUES,
                             OPEN_LOOP_VALUES)

"""
Signal generators for tremor estimation testing.

Provides functions to generate various synthetic tremor signals for
algorithm development and validation.
"""


def _generate_tremor_signal(duration: float,
                            fs: float,
                            frequency: float,
                            amplitude: float,
                            frequency_modulation: bool) -> np.ndarray:
    """
    Generate a synthetic tremor signal with optional frequency modulation.

    Parameters:
    -----------
    duration : float
        Signal duration in seconds
    fs : float
        Sampling frequency (Hz)
    frequency : float
        Fundamental tremor frequency (Hz)
    amplitude : float
        Tremor amplitude
    frequency_modulation : bool
        If True, tremor frequency varies sinusoidally between frequency ± 4 Hz

    Returns:
    --------
    t : ndarray
        Time vector (seconds)
    signal : ndarray
        Combined signal (tremor + voluntary + noise)
    tremor : ndarray
        Pure tremor component (without noise)
    voluntary : ndarray
        Pure voluntary motion component (low frequency)
    """

    t = np.arange(0, duration, 1/fs)

    if frequency_modulation:
        # Modulate frequency between frequency ± 4 Hz at 1 Hz
        # modulation rate
        freq_variation = frequency + \
            TREMOR_MODULATION_AMPLITUDE * np.sin(2 * np.pi * t)
        phase = 2 * np.pi * np.cumsum(freq_variation) / fs
    else:
        phase = 2 * np.pi * frequency * t

    # Generate sinusoidal tremor with 2nd harmonic
    tremor = amplitude * np.sin(phase) + 0.5 * amplitude * np.sin(2 * phase)

    return tremor


def _generate_measured_signal(**kwargs) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
    """
    Generate a signal with combined tremor and voluntary motion components.

    This simulates realistic surgical hand motion where involuntary tremor
    (higher frequency 4-12 Hz) is superimposed on voluntary motion (< 1 Hz).

    Parameters:
    -----------
    duration : float
        Signal duration in seconds
    fs : float
        Sampling frequency (Hz)
    tremor_freq : float
        Tremor component frequency in Hz (typically 4-12 Hz for physiology)
    tremor_amplitude : float
        Amplitude of tremor component
    voluntary_freq : float
        Voluntary motion frequency in Hz (must be <= 1 Hz)
    voluntary_amplitude : float
        Amplitude of voluntary motion component
    random_seed : int | None
        Random seed for noise generation. If None, noise will be different
        each time. Set to an integer for reproducible results.

    Returns:
    --------
    t : ndarray
        Time vector (seconds)
    signal : ndarray
        Combined signal: sin(w1*t) + sin(w2*t) with small noise
    tremor : ndarray
        Pure tremor component (without noise)
    voluntary : ndarray
        Pure voluntary motion component
    """
    np.random.seed(kwargs["random_seed"])

    if kwargs["voluntary_freq"] > 1.0:
        raise ValueError(
            f"Voluntary motion frequency must be <= 1 Hz, got "
            f"{kwargs['voluntary_freq']}"
        )

    t = np.arange(0, kwargs["duration"], 1/kwargs["fs"])

    # Voluntary motion component (low frequency, <= 1 Hz)
    voluntary = kwargs["voluntary_amplitude"] * \
        np.sin(2 * np.pi * kwargs["voluntary_freq"] * t)

    # Tremor component (physiological tremor, typically 4-12 Hz)
    tremor = _generate_tremor_signal(
        duration=kwargs["duration"],
        fs=kwargs["fs"],
        frequency=kwargs["tremor_base_freq"],
        amplitude=kwargs["tremor_amplitude"],
        frequency_modulation=kwargs["tremor_freq_modulation"]
    )

    # Additive noise (5% of tremor amplitude)
    noise = 0.05 * kwargs["tremor_amplitude"] * np.random.randn(len(t))

    # Combined signal
    signal = voluntary + tremor + noise

    np.savez(
        **kwargs,
        t=t,
        signal=signal,
        tremor=tremor,
        voluntary=voluntary,
    )

    return t, signal, tremor, voluntary


def _params_match(periodic_data, modulated_data) -> bool:
    """Validate that the parameters of the loaded signals match expected."""
    match = True

    ff_vals = list(FIXED_FREQUENCY_VALUES.keys())
    ff_vals.remove("file")

    mf_vals = list(MODULATED_FREQUENCY_VALUES.keys())
    mf_vals.remove("file")

    # Validate all parameters
    for param in ff_vals:
        if param not in periodic_data.files:
            print(f"Parameter '{param}' missing from loaded "
                  "fixed-frequency file.")
            return False

        if periodic_data[param] != FIXED_FREQUENCY_VALUES[param]:
            print(f"Parameter '{param}' mismatch in fixed-frequency data: "
                  f"expected {FIXED_FREQUENCY_VALUES[param]}, "
                  f"got {periodic_data[param]}")
            return False

    for param in mf_vals:
        if param not in modulated_data.files:
            print(f"Parameter '{param}' missing from loaded "
                  "frequency-modulated file.")
            return False

        if modulated_data[param] != MODULATED_FREQUENCY_VALUES[param]:
            print(f"Parameter '{param}' mismatch in frequency-modulated data: "
                  f"expected {MODULATED_FREQUENCY_VALUES[param]}, "
                  f"got {modulated_data[param]}")
            return False

    # If all checks passed, parameters are considered matched
    return match


def generate_example_signals():
    """
    Generate and save example signals for testing.
    """

    # Create output directory if it doesn't exist
    os.makedirs(INPUT_DIR, exist_ok=True)

    all_exist = all([os.path.exists(FIXED_FREQUENCY_VALUES["file"]),
                    os.path.exists(MODULATED_FREQUENCY_VALUES["file"]),
                    os.path.exists(OPEN_LOOP_VALUES["file"])])

    if all_exist:
        print("Signals already exist, validating parameters...")
        periodic_data = np.load(FIXED_FREQUENCY_VALUES["file"])
        modulated_data = np.load(MODULATED_FREQUENCY_VALUES["file"])
        op_data = np.load(OPEN_LOOP_VALUES["file"])

        # Validate that cached signals have matching parameters
        if _params_match(periodic_data, modulated_data):
            print("Parameters match, loading...")
            return {
                FIXED_FREQUENCY_NPZ: (
                    periodic_data['t'],
                    periodic_data['signal'],
                    periodic_data['tremor'],
                    periodic_data['voluntary']
                ),
                MODULATED_FREQUENCY_NPZ: (
                    modulated_data['t'],
                    modulated_data['signal'],
                    modulated_data['tremor'],
                    modulated_data['voluntary']
                ),
                OPEN_LOOP_VALUES["file"]: (
                    op_data['t'],
                    op_data['theta_noisy'],
                    op_data['theta_tremor'],
                    op_data['theta_voluntary']
                )
            }
        else:
            print("Parameters do not match previous values.")
    print(f"Generating example signals in '{INPUT_DIR}'...")

    # Signal 1: Fixed-frequency tremor + voluntary motion
    t1, sig1, tremor1, voluntary1 = _generate_measured_signal(
        **FIXED_FREQUENCY_VALUES)
    print("\t* Saved: periodic_tremor.npz")

    # Signal 2: Frequency-modulated tremor + voluntary motion
    t2, sig2, tremor2, voluntary2 = _generate_measured_signal(
        **MODULATED_FREQUENCY_VALUES)
    print("\t* Saved: modulated_tremor.npz")

    # Signal 3: Open loop response (tremor + voluntary + noise, no modulation)
    op_data = np.load(OPEN_LOOP_VALUES["file"])
    t3, sig3, tremor3, voluntary3 = (op_data['t'],
                                     op_data['theta_noisy'],
                                     op_data['theta_tremor'],
                                     op_data['theta_voluntary'])
    print("\t* Loaded: open_loop_response.npz")

    return {
        "periodic_tremor.npz": (t1, sig1, tremor1, voluntary1),
        "modulated_tremor.npz": (t2, sig2, tremor2, voluntary2),
        "open_loop_response.npz": (t3, sig3, tremor3, voluntary3)
    }


if __name__ == "__main__":
    generate_example_signals()
