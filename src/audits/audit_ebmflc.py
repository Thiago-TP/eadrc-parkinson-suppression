"""
Audit of the EBMFLC voluntary-motion estimate of EADRC+EBMFLC
(src/control_strategies/eadrc_ebmflc.py).

The estimate must sum both the sine and the cosine terms of the reference
frequencies up to 4 Hz,
    v_hat = sum_{f_r <= 4 Hz} (w_r sin(w_r t) + w_{n+r} cos(w_r t)),
where x_m = [sin(ws t), cos(ws t)] holds the sine of frequency r at r and its
cosine at n + r. Up to 2026-10-08 the code indexed x_m with
`voluntary_indices * 2`, which repeats the Python list instead of adding the
offset n, so the estimate was 2 * sum(w_r sin(w_r t)) and ignored the cosines.

Checks, each reported as PASS or FAIL (exit code 1 on any failure):
1. the voluntary terms are exactly the sine and cosine of each frequency up to 4 Hz;
2. on the stored nominal runs (results/runs), replaying the estimator on the
   stored theta_3 reproduces the stored estimate, which matches the reference formula;
3. on a synthetic signal with known voluntary and tremor parts, the estimate
   matches the reference formula; the error of the old sine-only estimate is
   shown for comparison.

Run from the repository root:
    uv run src/audits/audit_ebmflc.py
"""

import contextlib
import io
import pickle
import sys
from pathlib import Path

import blosc
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control_strategies.eadrc_ebmflc import EADRC_EBMFLC  # noqa: E402
from system import ModelParameters  # noqa: E402

TOLERANCE = 1e-12  # [rad]
failures = 0


def check(passed: bool, message: str) -> None:
    global failures
    failures += not passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {message}")


def new_estimator() -> EADRC_EBMFLC:
    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        return EADRC_EBMFLC(
            name="eadrc_ebmflc",
            params=ModelParameters(**cfgs["parameters"]),
            ic=tuple(cfgs["initial_conditions"].values()),
            amplitude_voluntary=1.0,
        )


def replay(est: EADRC_EBMFLC, theta3: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Feeds theta_3 to the estimator one step at a time, as System.simulate_system
    does (k = 1, ..., N - 1), and evaluates, with the very weights the estimator
    uses at each step, the reference estimate and the old sine-only one.
    The reference frequencies are taken from the grid itself (f_r <= 4 Hz),
    not from the estimator's index lists.
    """
    voluntary = np.flatnonzero(est.ws / (2 * np.pi) <= 4 + 1e-9)
    sines, cosines = voluntary, voluntary + est.n
    estimate, reference, sine_only = (np.zeros(len(theta3)) for _ in range(3))
    for k in range(1, len(theta3)):
        wt = est.ws[voluntary] * k * est.dt
        w = est.w_m.copy()  # weights used at step k, before their update
        reference[k] = w[sines] @ np.sin(wt) + w[cosines] @ np.cos(wt)
        sine_only[k] = 2 * w[sines] @ np.sin(wt)
        est.theta[k, 2] = theta3[k]
        est._update_estimates(k)
        estimate[k] = est.theta_v_hat[k, 2]
    return estimate, reference, sine_only


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


est = new_estimator()
n = est.n
grid = est.ws / (2 * np.pi)
expected = set(np.flatnonzero(grid <= 4 + 1e-9)) | {i + n for i in np.flatnonzero(grid <= 4 + 1e-9)}

print("1. Selected terms of x_m = [sin(ws t), cos(ws t)]")
terms = list(getattr(est, "voluntary_terms", []))
check(len(terms) > 0, "the estimator lists its voluntary terms (voluntary_terms)")
check(len(terms) == len(set(terms)), f"no term selected twice ({len(terms)} terms)")
check(set(terms) == expected, f"sine and cosine of the {len(expected) // 2} frequencies up to 4 Hz")

print("2. Stored nominal EADRC+EBMFLC runs (results/runs), estimator replayed on the stored theta_3")
for amplitude in ("0.0", "1.0"):
    path = Path(f"results/runs/eadrc_ebmflc_amplitude_{amplitude}.data")
    if not path.exists():
        check(False, f"{path} not found (run `uv run src/main.py` first)")
        continue
    with open(path, "rb") as f:
        run = pickle.loads(blosc.decompress(f.read()))["nominal_run"]
    stored = run["theta_v_hat"][:, 2]
    estimate, reference, _ = replay(new_estimator(), run["theta"][:, 2])
    # k = 0 is skipped: theta_v_hat[0] is initialized to theta[0] before the estimator runs
    check(np.abs(estimate - stored)[1:].max() <= TOLERANCE,
          f"amplitude {amplitude}: replay reproduces the stored estimate")
    check(np.abs(stored - reference)[1:].max() <= TOLERANCE,
          f"amplitude {amplitude}: stored estimate matches the reference formula")

print("3. Synthetic signal: voluntary 0.3 Hz with sine and cosine parts, tremor 6 Hz")
t = np.arange(0, 6 + 1e-3, 1e-3)
voluntary = 0.2 * np.sin(2 * np.pi * 0.3 * t) + 0.2 * np.cos(2 * np.pi * 0.3 * t)
tremor = 0.1 * np.sin(2 * np.pi * 6 * t)
estimate, reference, sine_only = replay(new_estimator(), voluntary + tremor)
check(np.abs(estimate - reference).max() <= TOLERANCE, "estimate matches the reference formula")
late = t >= 3  # after the adaptation transient
print(f"  RMS error vs the true voluntary part for t >= 3 s: {rms(estimate[late] - voluntary[late]):.4f} rad "
      f"(old sine-only estimate: {rms(sine_only[late] - voluntary[late]):.4f} rad; "
      f"signal RMS: {rms(voluntary[late]):.4f} rad)")

print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")
sys.exit(failures > 0)
