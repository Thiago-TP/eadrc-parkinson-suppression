"""
Spectrograms of the wrist response on the nominal model, from long simulations.

The Monte Carlo runs last 6 s, too short for a fine frequency resolution, so
this script simulates the nominal model for 1000 s (frequency resolution of
1 mHz) for the uncontrolled system and for EADRC+EBMFLC, at rest and in
deliberate motion, and saves their spectrograms with
Plots.plot_spectrogram_response to results/plots/<strategy>_amplitude_<amplitude>/.

Run from the repository root:
    uv run src/postprocessing/spectrograms.py
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control_strategies.eadrc_ebmflc import EADRC_EBMFLC  # noqa: E402
from control_strategies.uncontrolled import Uncontrolled  # noqa: E402
from plots import Plots  # noqa: E402
from system import System, ModelParameters  # noqa: E402

DURATION = 1000.0  # [s]
STRATEGIES = {"uncontrolled": Uncontrolled, "eadrc_ebmflc": EADRC_EBMFLC}
AMPLITUDES = (0.0, 1.0)


def extend_duration(control: System, t1: float) -> None:
    """Resizes the time vector and the histories of a fresh simulation to end at t1."""
    control.t1 = t1
    control.t = np.arange(control.t0, control.t1 + control.dt, control.dt)
    for name, width in [
        ("u", 3), ("x", 6), ("x_v", 6), ("theta", 3), ("theta_v", 3),
        ("theta_v_hat", 3), ("theta_i", 3), ("theta_i_hat", 3),
    ]:
        setattr(control, name, np.zeros((len(control.t), width)))
    control._initialize_simulation_attributes()


def nominal_long_run(job: tuple[str, float]) -> dict:
    strategy, amplitude = job
    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)
    control = STRATEGIES[strategy](
        name=strategy,
        params=ModelParameters(**cfgs["parameters"]),
        ic=tuple(cfgs["initial_conditions"].values()),
        amplitude_voluntary=amplitude,
    )
    extend_duration(control, DURATION)
    control.simulate_system()
    return control.results["nominal_run"]


if __name__ == "__main__":
    jobs = [(strategy, amplitude) for strategy in STRATEGIES for amplitude in AMPLITUDES]
    with Pool(len(jobs)) as pool:
        runs = dict(zip(jobs, pool.map(nominal_long_run, jobs)))

    for (strategy, amplitude), run in runs.items():
        Plots(
            control_name=strategy,
            theta_baseline=runs[("uncontrolled", amplitude)]["theta"],
            **run,
        ).plot_spectrogram_response()
