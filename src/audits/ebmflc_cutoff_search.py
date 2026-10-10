"""
Grid search of the voluntary-motion cutoff of the EBMFLC estimator in EADRC+EBMFLC
(src/control_strategies/eadrc_ebmflc.py).

EBMFLC splits its reference frequencies (0-12 Hz) at a cutoff f_v: the terms up
to f_v form the voluntary-motion estimate fed to the EADRC, the rest the tremor
estimate. This script simulates EADRC+EBMFLC in closed loop for
f_v in {0, 0.5, ..., 4} Hz, at rest and in deliberate motion, and ranks the
cutoffs by the R^2 of theta_3 with respect to theta_v3, the same criterion used
to tune the PIDs (src/pid_tuning.py); TPSR, IAE, entropy, control power and
total variation are reported too, from src/postprocessing/metrics.py.

By default only the nominal model is simulated; --monte-carlo adds the 99
stiffness samples of configs.yaml (100 runs per cutoff and scenario, averaged).
Simulations run in parallel, one process per CPU core.

Check, reported as PASS or FAIL (exit code 1 on failure): the 4 Hz cutoff,
used in the study, reproduces the stored nominal runs (results/runs).

Results are printed and saved to results/metrics/ebmflc_cutoff_search.csv.

Run from the repository root:
    uv run src/audits/ebmflc_cutoff_search.py
    uv run src/audits/ebmflc_cutoff_search.py --monte-carlo
"""

import argparse
import contextlib
import io
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import blosc
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control_strategies.eadrc_ebmflc import EADRC_EBMFLC  # noqa: E402
from control_strategies.uncontrolled import Uncontrolled  # noqa: E402
from postprocessing.metrics import _compute_metrics  # noqa: E402
from system import ModelParameters, System  # noqa: E402

CUTOFFS = np.arange(0.0, 4.0 + 1e-9, 0.5)  # [Hz]
STUDY_CUTOFF = 4.0  # [Hz], EADRC_EBMFLC default
TOTAL_BAND = (0.0, 12.0)  # [Hz], EADRC_EBMFLC default
AMPLITUDES = (0.0, 1.0)  # rest, deliberate motion
METRICS = ["r2", "tpsr_percent", "iae", "response_entropy", "control_power", "control_tvc"]
OUTPUT = Path("results/metrics/ebmflc_cutoff_search.csv")
TOLERANCE = 1e-12  # [rad]


def simulate(control: System, sample: int | None) -> dict:
    """Simulates one run: the nominal model (sample=None) or a row of the stiffness table."""
    with contextlib.redirect_stdout(io.StringIO()):
        if sample is not None:
            control.set_stiffness_sample(sample)
        control.simulate_system()
    return next(iter(control.results.values()))


def new_control(cls: type, amplitude: float, **kwargs) -> System:
    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(
            name=cls.__name__.lower(),
            params=ModelParameters(**cfgs["parameters"]),
            ic=tuple(cfgs["initial_conditions"].values()),
            amplitude_voluntary=amplitude,
            **kwargs,
        )


def baseline_job(job: tuple[float, int | None]) -> tuple[tuple[float, int | None], dict]:
    amplitude, sample = job
    run = simulate(new_control(Uncontrolled, amplitude), sample)
    return job, {"theta": run["theta"], "theta_v": run["theta_v"]}


def ebmflc_job(job: tuple[float, float, int | None, dict]) -> dict:
    cutoff, amplitude, sample, baseline = job
    control = new_control(
        EADRC_EBMFLC,
        amplitude,
        total_bandwidth=TOTAL_BAND,
        voluntary_bandwidth=(TOTAL_BAND[0], cutoff),
        tremor_bandwidth=(cutoff, TOTAL_BAND[1]),
    )
    run = simulate(control, sample)
    metrics = _compute_metrics(run_payload=run, baseline_payload=baseline)
    row = {"cutoff_hz": cutoff, "amplitude": amplitude, "sample": -1 if sample is None else sample}
    row.update({m: metrics[m] for m in METRICS})
    if cutoff == STUDY_CUTOFF and sample is None:
        row["theta"] = run["theta"]  # kept for the reproduction check
    return row


def reproduction_check(rows: list[dict]) -> bool:
    print("Check: the 4 Hz cutoff reproduces the stored nominal EADRC+EBMFLC runs")
    passed = True
    for amplitude in AMPLITUDES:
        path = Path(f"results/runs/eadrc_ebmflc_amplitude_{amplitude}.data")
        if not path.exists():
            print(f"  [SKIP] {path} not found (run `uv run src/main.py` first)")
            continue
        with open(path, "rb") as f:
            stored = pickle.loads(blosc.decompress(f.read()))["nominal_run"]["theta"]
        row = next(r for r in rows if "theta" in r and r["amplitude"] == amplitude)
        ok = np.abs(row["theta"] - stored).max() <= TOLERANCE
        passed &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] amplitude {amplitude}")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--monte-carlo", action="store_true",
                        help="also simulate the 99 stiffness samples of configs.yaml")
    args = parser.parse_args()

    with open("configs.yaml") as f:
        samples = [None] + (
            list(range(len(yaml.safe_load(f)["parameters"]["stiffness_samples"])))
            if args.monte_carlo else []
        )

    with Pool() as pool:
        baselines = dict(pool.map(baseline_job, [(a, s) for a in AMPLITUDES for s in samples]))
        jobs = [(c, a, s, baselines[(a, s)]) for c in CUTOFFS for a in AMPLITUDES for s in samples]
        print(f"Simulating {len(jobs)} EADRC+EBMFLC runs "
              f"({len(CUTOFFS)} cutoffs x {len(AMPLITUDES)} scenarios x {len(samples)} models)...")
        rows = pool.map(ebmflc_job, jobs, chunksize=1)

    passed = reproduction_check(rows)

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in rows])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)

    label = "mean over nominal + 99 samples" if args.monte_carlo else "nominal model"
    pd.set_option("display.width", 160)
    for amplitude, scenario in zip(AMPLITUDES, ("rest", "deliberate motion")):
        summary = table[table.amplitude == amplitude].groupby("cutoff_hz")[METRICS].mean()
        best = summary["r2"].idxmax()
        print(f"\n{scenario.capitalize()} ({label}); best cutoff by R^2: {best:.1f} Hz"
              f" (study: {STUDY_CUTOFF:.1f} Hz)")
        print(summary.round(4).to_string())
    print(f"\nAll runs saved to {OUTPUT}")
    sys.exit(not passed)
