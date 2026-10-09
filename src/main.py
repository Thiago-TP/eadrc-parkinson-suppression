import yaml

from control_strategies import (
    afe_notch,
    eadrc_ebmflc,
    eadrc_zplp,
    pi_gallego,
    pid,
    uncontrolled,
)
from system import ModelParameters


def main(
    num_simulations: int,
    amplitude_voluntary: float = 1.0,
    strategies: list[str] | None = None,
) -> None:
    """
    Main function to run and persist simulations.

    Parameters
    ----------
    num_simulations : int
        The number of simulations to run.
        The first run is always the nominal model, and remaining runs
        take the rows of the stiffness samples table in order, the same
        for every control strategy, so at most len(table) + 1 runs.
        A properly formatted configs.yaml file is required to specify
        nominal parameters and the stiffness samples table.
    amplitude_voluntary : float, optional
        Amplitude of the voluntary torque profile.
        Defaults to 1.0 (the more interesting case).
    strategies : list[str], optional
        Names of the control strategies to simulate.
        Defaults to None, which simulates all of them.
    """

    # Load configurations
    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)

    # Load nominal model parameters
    parameters = ModelParameters(**cfgs["parameters"])

    num_samples = len(parameters.stiffness_samples)
    if num_simulations - 1 > num_samples:
        raise ValueError(
            f"{num_simulations} simulations need {num_simulations - 1} "
            f"stiffness samples, but configs.yaml has {num_samples}."
        )

    # Load initial conditions
    ic = tuple(cfgs["initial_conditions"].values())

    # Instantiate control strategies
    # with same model parameters and initial conditions
    afe_notch_control = afe_notch.AFE_NotchControl(
        name="afe_notch",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
    )
    eadr_ebmflc_control = eadrc_ebmflc.EADRC_EBMFLC(
        name="eadrc_ebmflc",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
    )
    eadr_zplp_control = eadrc_zplp.EADRC_ZPLP(
        name="eadrc_zplp",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
    )
    pi_gallego_control = pi_gallego.GallegoPIControl(
        name="pi_gallego",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
    )
    pid_imc_control = pid.PIDControl(
        name="pid_imc",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
        # Values below were found by the grid search of src/pid_tuning.py
        # (tune_flawed_tracker) on the nominal model, in both scenarios
        manual=True,
        # IMC gains for slow_factor=3.9, grid search on the 2026-05-01 code
        # (null initial state, R2 normalized by the variance of theta_3)
        # kp=0.1266318,
        # ki=126.6317684,
        # kd=0.0519192,
        # IMC gains for slow_factor=0.4 (best from grid search, current code)
        kp=1.2346597,
        ki=1234.6597421,
        kd=0.5062123,
    )
    pid_de_control = pid.PIDControl(
        name="pid_de",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
        # Values below were found by the DE of src/pid_tuning.py
        # (tune_perfect_tracker) on the nominal model, in both scenarios
        manual=True,
        # Perfect tracker gains from DE on the 2026-05-01 code
        # (null initial state, R2 normalized by the variance of theta_3)
        # kp=2.8024576,
        # ki=16.3107364,
        # kd=3.2077601,
        # Perfect tracker gains from DE (current code)
        kp=1.7212542,
        ki=15.0591711,
        kd=3.2392269,
        perfect_tracking=True,
    )
    no_control = uncontrolled.Uncontrolled(
        name="uncontrolled",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
    )

    # Run nominal model simulations for selected control strategies
    print("\nRunning nominal model simulations...")
    controls = [
        afe_notch_control,
        eadr_ebmflc_control,
        eadr_zplp_control,
        pi_gallego_control,
        pid_de_control,
        pid_imc_control,
        no_control,
    ]
    if strategies is not None:
        controls = [control for control in controls if control.name in strategies]
    for control in controls:
        control.simulate_system()

    # Run non-nominal model simulations with stiffness sampling
    # for selected control strategies
    print(
        f"\nRunning {num_simulations - 1} "
        "non-nominal model simulations with tabulated stiffness samples..."
    )
    for index in range(num_simulations - 1):
        for control in controls:
            # Same table row for every control -> runs are paired
            control.set_stiffness_sample(index)
            control.simulate_system()

    # Save results across runs to npz files in results folder
    for control in controls:
        control.save_results()


if __name__ == "__main__":
    import time

    # Control strategies compared in the CBA 2026 paper
    paper_strategies = [
        "eadrc_ebmflc",
        "eadrc_zplp",
        "pid_de",
        "pid_imc",
        "uncontrolled",
    ]

    __start = time.time()
    main(
        num_simulations=100,
        amplitude_voluntary=0.0,
        strategies=paper_strategies,
    )
    main(
        num_simulations=100,
        amplitude_voluntary=1.0,
        strategies=paper_strategies,
    )
    __stop = time.time()

    delta_s = __stop - __start
    delta_m = delta_s / 60
    print(f"\nAll finished in {delta_s:.2f}s ({delta_m:.2f} minutes)")
