# Source package overview

This directory contains the executable simulation and analysis code for the tremor suppression project. It is the main development area for the model, controllers, and post-processing pipeline.

## Entry points and core modules

- [main.py](main.py) — simulation entry point; loads [../configs.yaml](../configs.yaml), instantiates controllers, runs the nominal model and the tabulated stiffness samples (the same for every controller), and saves results
- [system.py](system.py) — model parameters and dynamics used by all control strategies
- [pid_tuning.py](pid_tuning.py) — PID tuning utilities and parameter search scripts
- [requirements.txt](requirements.txt) — pip-style list of dependencies (the uv environment, from [../pyproject.toml](../pyproject.toml) and [../uv.lock](../uv.lock), is the reference one)

## Control strategy implementations

The [control_strategies](control_strategies) package contains the controllers used during simulation:

- [control_strategies/afe_notch.py](control_strategies/afe_notch.py) — adaptive notch filtering approach
- [control_strategies/eadrc_ebmflc.py](control_strategies/eadrc_ebmflc.py) — EADRC controller with EBMFLC logic
- [control_strategies/eadrc_zplp.py](control_strategies/eadrc_zplp.py) — EADRC controller with ZPLP design
- [control_strategies/pi_gallego.py](control_strategies/pi_gallego.py) — Gallego PI controller
- [control_strategies/pid.py](control_strategies/pid.py) — PID implementation used for tuning and comparison
- [control_strategies/uncontrolled.py](control_strategies/uncontrolled.py) — uncontrolled baseline reference

## Post-processing and analysis

The [postprocessing](postprocessing) package converts saved simulation results into plots and metrics:

- [postprocessing/postprocess.py](postprocessing/postprocess.py) — top-level pipeline to generate tables and plots
- [postprocessing/metrics.py](postprocessing/metrics.py) — metric calculations and CSV writing utilities
- [postprocessing/plots.py](postprocessing/plots.py) — plotting routines
- [postprocessing/statistics.py](postprocessing/statistics.py) — summary/statistical helpers
- [postprocessing/spectrograms.py](postprocessing/spectrograms.py) — spectrograms of the nominal model from 1000 s simulations (uncontrolled and EADRC+EBMFLC), for a fine frequency resolution
- [postprocessing/document_tables.py](postprocessing/document_tables.py) — updates the numbers of the results tables of the CBA 2026 paper and presentation from the post-processed results

## Audits

The [audits](audits) folder holds scripts that check specific behaviors of the code, each reporting PASS or FAIL and exiting with a nonzero code on failure:

- [audits/audit_ebmflc.py](audits/audit_ebmflc.py) — checks that the EBMFLC voluntary-motion estimate of EADRC+EBMFLC sums both the sine and the cosine terms up to 4 Hz (until 2026-10-08, the cosine terms were dropped), on the stored runs and on a synthetic signal
- [audits/ebmflc_cutoff_search.py](audits/ebmflc_cutoff_search.py) — grid search (0 to 4 Hz, 0.5 Hz steps, in parallel) of the voluntary-motion cutoff of EBMFLC in closed loop, ranked by $R^2$, on the nominal model or, with `--monte-carlo`, on the 100 runs of the stiffness table; also checks that the 4 Hz cutoff of the study reproduces the stored runs

## Tremor estimation methods

The [tremor_estimation_strategies](tremor_estimation_strategies) folder contains the literature benchmark and algorithm comparison components:

- [tremor_estimation_strategies/methods](tremor_estimation_strategies/methods) — estimator implementations
- [tremor_estimation_strategies/results](tremor_estimation_strategies/results) — per-method result folders
- [tremor_estimation_strategies/utils](tremor_estimation_strategies/utils) — constants, logging, plotting, and signal helpers
- [tremor_estimation_strategies/input_examples](tremor_estimation_strategies/input_examples) — example tremor signals
- [tremor_estimation_strategies/literature_review](tremor_estimation_strategies/literature_review) — papers and review artifacts
- [tremor_estimation_strategies/run_methods.py](tremor_estimation_strategies/run_methods.py) — execute the estimation methods
- [tremor_estimation_strategies/table_results.py](tremor_estimation_strategies/table_results.py) — aggregate comparison tables

## Typical workflow

From the repo root (paths to [../configs.yaml](../configs.yaml) and [../results](../results) are relative to the working directory):

```bash
uv sync
uv run src/main.py
uv run src/postprocessing/postprocess.py
uv run src/postprocessing/spectrograms.py
uv run src/postprocessing/document_tables.py
```

This mirrors the project workflow used in the repository:

1. load the configuration file
2. simulate the nominal model and the perturbed models given by the table of stiffness samples
3. save the numerical outputs in the results directory
4. generate plots and summary metrics from those outputs

## Important notes

- The project is structured as a research simulation workflow, not as a packaged library.
- Imports resolve modules from the folder of the script being run, while file paths are relative to the repository root, which must be the working directory.
- The simulation and post-processing steps depend on the configuration in [../configs.yaml](../configs.yaml) and the outputs in [../results](../results).