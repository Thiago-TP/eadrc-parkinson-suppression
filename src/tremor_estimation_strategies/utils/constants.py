import os

"""
Constants and parameters for signal generation.
"""

INPUT_DIR = "input_examples"
FIXED_FREQUENCY_NPZ = "periodic_tremor.npz"
MODULATED_FREQUENCY_NPZ = "modulated_tremor.npz"

TREMOR_FREQUENCY = 8.0  # tremor frequency for fixed-frequency signal (Hz)
TREMOR_MODULATION_AMPLITUDE = 4.0
TREMOR_AMPLITUDE = 1.0  # amplitude of the tremor component
MIN_FREQ = TREMOR_FREQUENCY - TREMOR_MODULATION_AMPLITUDE
MAX_FREQ = TREMOR_FREQUENCY + TREMOR_MODULATION_AMPLITUDE

VOLUNTARY_FREQUENCY = 0.5  # Hz, typical voluntary motion frequency
VOLUNTARY_AMPLITUDE = 1.0  # amplitude of the voluntary component

DURATION = 3.0  # seconds
FS = 1000.0  # sampling frequency (Hz)

SEEDS = {
    "periodic": 42,
    "modulated": 43
}

COMMON_VALUES = {
    "duration": DURATION,
    "fs": FS,
    "voluntary_amplitude": VOLUNTARY_AMPLITUDE,
    "voluntary_freq": VOLUNTARY_FREQUENCY,
    "tremor_base_freq": TREMOR_FREQUENCY,
    "tremor_amplitude": TREMOR_AMPLITUDE,
}

FIXED_FREQUENCY_VALUES = COMMON_VALUES | {
    "file": os.path.join(INPUT_DIR, FIXED_FREQUENCY_NPZ),
    "tremor_freq_modulation": False,
    "random_seed": SEEDS['periodic'],
    "description": (f"Fixed-frequency {TREMOR_FREQUENCY} Hz tremor + "
                    f"{VOLUNTARY_FREQUENCY} Hz voluntary motion")
}

MODULATED_FREQUENCY_VALUES = COMMON_VALUES | {
    "file": os.path.join(INPUT_DIR, MODULATED_FREQUENCY_NPZ),
    "tremor_freq_modulation": True,
    "random_seed": SEEDS['modulated'],
    "description": (f"Modulated-frequency {MIN_FREQ}-{MAX_FREQ} Hz tremor + "
                    f"{VOLUNTARY_FREQUENCY} Hz voluntary motion")
}

OPEN_LOOP_NPZ = "open_loop_response.npz"
OPEN_LOOP_VALUES = {
    "file": os.path.join(INPUT_DIR, OPEN_LOOP_NPZ),
    "description": "Open loop response (tremor + voluntary + noise, no modulation)"
}
