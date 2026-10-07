# EADRC Parkinson's Tremor Suppression

This repository studies tremor suppression in a 3-DOF biomechanical arm model by comparing several control strategies under parameter uncertainty. The codebase is organized around a simulation workflow: define the model, run multiple stochastic realizations, and then summarize the resulting trajectories with post-processing metrics and plots.

## Project goals

- Model a human arm as a 3-joint system with nominal parameters and sampled stiffness uncertainty
- Compare multiple suppression strategies against the same baseline conditions
- Save simulation outputs to a reproducible results directory
- Generate summary plots and metrics tables after the simulations finish

## Repository layout

```text
.
├── LICENSE
├── README.md
├── configs.yaml
├── pyproject.toml
├── results/
│   ├── metrics/
│   ├── plots/
│   └── runs/
├── cba-2026-paper/
├── cba-2026-presentation/
├── docs/
│   └── literature_review/
└── src/
    ├── README.md
    ├── main.py
    ├── pid_tuning.py
    ├── requirements.txt
    ├── system.py
    ├── control_strategies/
    │   ├── afe_notch.py
    │   ├── eadrc_ebmflc.py
    │   ├── eadrc_zplp.py
    │   ├── pi_gallego.py
    │   ├── pid.py
    │   └── uncontrolled.py
    ├── postprocessing/
    │   ├── metrics.py
    │   ├── plots.py
    │   ├── postprocess.py
    │   └── statistics.py
    └── tremor_estimation_strategies/
        ├── input_examples/
        ├── literature_review/
        ├── methods/
        ├── results/
        ├── run_methods.py
        ├── table_results.py
        └── utils/
```

## Model and configuration

The nominal arm parameters and uncertainty ranges are defined in [configs.yaml](configs.yaml). This file contains:

- geometric and inertial parameters for the shoulder, elbow, and wrist dynamics
- stiffness values and stiffness intervals for Monte Carlo-style sampling
- a table of 99 stiffness samples (`stiffness_samples`), drawn once from those intervals: non-nominal run *i* uses row *i* for every control strategy and scenario, so runs are paired across controllers and the results do not depend on how many strategies are simulated, or in which order
- initial conditions for the joint states

The main model logic is implemented in [src/system.py](src/system.py), and the top-level simulation driver is [src/main.py](src/main.py).

## Control strategies in the current codebase

The active simulation entry point currently instantiates and runs multiple controller variants, including:

- AFE notch filtering
- EADRC with EBMFLC
- EADRC with ZPLP
- Gallego PI
- PID controllers
- uncontrolled baseline

The exact controller implementations live under [src/control_strategies](src/control_strategies).

## Getting started

The environment is managed with [uv](https://docs.astral.sh/uv/): [pyproject.toml](pyproject.toml) declares the dependencies, [uv.lock](uv.lock) pins their exact versions and [.python-version](.python-version) the Python version.

### 1. Create the environment

From the repository root, create the `.venv` folder with the locked dependencies (uv downloads the required Python if needed):

```bash
uv sync
```

### 2. Run the simulations

Run the simulation entry point from the repository root, since it reads [configs.yaml](configs.yaml) and writes to [results/runs](results/runs) relative to the working directory:

```bash
uv run src/main.py
```

This runs the control strategies compared in the CBA 2026 paper on the nominal model and on the 99 tabulated stiffness samples, for both scenarios (rest and deliberate motion). Results are written under [results/runs](results/runs).

## Output files

The workflow writes numerical outputs and then post-processes them.

### Simulation outputs

The generated run files live in [results/runs](results/runs). These are the primary numeric artifacts used to compare controllers and amplitudes.

### Post-processing

From the repository root, run:

```bash
uv run src/postprocessing/postprocess.py
```

This script reads the saved run files and generates summary plots and metrics under:

- [results/plots](results/plots)
- [results/metrics](results/metrics)

## Notes on the workflow

- The simulation driver in [src/main.py](src/main.py) accepts `num_simulations`, `amplitude_voluntary` and `strategies` (names of the control strategies to simulate; all of them by default) parameters.
- The configuration file drives both the nominal model and the table of stiffness samples for robustness analysis.
- This repository is organized as a research/simulation project rather than a packaged application, so the source directory is the primary execution context.

## CBA 2026 paper and presentation

This work is the subject of the paper *Error-Based Active Disturbance Rejection Control for Parkinson's Disease Tremor Suppression in Wrist Considering Upper Limb Dynamics*, presented at the XXVI Congresso Brasileiro de Automática (CBA 2026, the Brazilian Congress on Automation), held in São Paulo, Brazil, on October 6–9, 2026. More information about the conference is available at [sites.usp.br/cba2026](https://sites.usp.br/cba2026/).

- [cba-2026-paper](cba-2026-paper) holds the LaTeX sources of the paper, written with the `ifacconf` class of the conference template: sections, tables, figures, references, and the reviewers' comments with the authors' answers to them (in Portuguese).
- [cba-2026-presentation](cba-2026-presentation) holds the Beamer slides of the 15-minute oral presentation, in Portuguese, built on the official CBA 2026 template (`cba2026.sty`). The slides reuse the paper's figures, upper limb schematic and bibliography straight from [cba-2026-paper](cba-2026-paper), so both folders must be kept side by side.

Both documents are built with a TeX distribution that provides `pdflatex`, `bibtex` and `biber` (e.g. TeX Live), from Git Bash or any other bash shell:

```bash
bash cba-2026-paper/build.sh          # writes cba-2026-paper/paper.pdf
bash cba-2026-presentation/build.sh   # writes cba-2026-presentation/presentation.pdf
```

Each script reruns `pdflatex` until citations and cross-references settle, prints any errors or warnings left in the log (no output means a clean build), and removes the auxiliary files.

## Related documentation

- [src/README.md](src/README.md) describes the source tree in more detail.
- [docs/literature_review](docs/literature_review) contains background material and review artifacts.

## License

This project is licensed under the [LICENSE](LICENSE) file.