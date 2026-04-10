"""
Tremor Estimation Methods
Methods for estimating and modeling physiological human tremor.
"""

import os
import numpy as np
from methods.algorithms import (
    # highpass_filter,
    # lowpass_filter,
    # flc,
    # bmflc,
    # ebmflc,
    # wflc,
    # abpf,
    # bbf,
    # cdf,
    # kalman_filter,
    # wflc_kf,
    # bmflc_rls,
    # bmflc_kf,
    # ar_lms,
    # ar_kf,
    # as_bmflc,
    # zpafkf,
    # amolc,
    ehwflc_kf,
)
from utils.signal_generator import generate_example_signals
from utils.plotting import plot_demonstration_results
from utils.method_parameters import get_method_parameters
from utils import logging


def demonstrate_method(method_class,
                       method_name: str,
                       input_signals: dict,
                       method_source: str,
                       output_base: str = "results",
                       verbose: bool = False):
    """
    Generic method for demonstrating tremor estimation algorithms.

    All methods estimated are treated as tremor estimates. Voluntary motion is
    reconstructed as: voluntary_estimate = signal - tremor_estimate.

    Parameters:
    -----------
    method_class : class
        Tremor estimation method class (e.g., WeightedFourierLinearCombiner)
    method_name : str
        Name of the method for display
    input_signals : dict
        Dictionary of test signals which values are tuple of
        (t, signal, tremor_true, voluntary_true)
    method_source : str
        Source module name for result organization (e.g., "wflc", "abpf")
    verbose : bool
        Whether to print logging information (default: False)
    """
    if verbose:
        logging.print_method_header(method_name)

    # Get method parameters from centralized configuration
    method_params = get_method_parameters(method_name)
    print(f"\nParameters for {method_name}: {method_params}")

    # Run tests in a loop
    results = {}
    for case_study, input_data in input_signals.items():

        # Unpack signal data
        if len(input_data) != 4:
            raise ValueError(
                "Expected signal data to contain 4 elements "
                "(t, signal, true_tremor, true_voluntary), "
                f"but got {len(input_data)}"
            )
        t, sig, true_tremor, true_voluntary = input_data

        # Initialize/reset method instance before each test case
        method_instance = method_class(**method_params)

        # Estimate tremor and voluntary motion
        run_output = method_instance.run(sig)
        tremor_estimates = run_output.tremor_estimates
        voluntary_estimates = run_output.voluntary_estimates

        results[case_study] = {
            "t": t,
            "signal": sig,
            "true_tremor": true_tremor,
            "tremor_estimates": tremor_estimates,
            "true_voluntary": true_voluntary,
            "voluntary_estimates": voluntary_estimates,
        }

    # Plot results
    plot_demonstration_results(results, method_name, method_source)

    # Save results to .npz file
    output_dir = os.path.join(output_base, method_source)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{method_source}_results.npz"
    output_path = os.path.join(output_dir, filename)
    np.savez_compressed(output_path, **results)

    return results


def main(verbose: bool = False):
    """
    Main demonstration script.

    Generates example input signals and demonstrates all
    tremor/voluntary movement estimation methods.
    Focus is on voluntary motion reconstruction quality.

    Parameters:
    -----------
    verbose : bool
        Whether to print logging information (default: False)
    """
    if verbose:
        logging.print_start_message()

    # Generate example signals
    signal_data = generate_example_signals()

    # Prepare input signals for demonstration
    input_signals = {k: v for k, v in zip(
        ["Fixed Frequency Tremor",
         "Modulated Frequency Tremor",
         "Open loop response"],
        signal_data.values()
    )}

    # Define methods to demonstrate
    # Each entry: (class, display_name, source_folder)
    methods_to_demonstrate = [
        # (
        #     highpass_filter.HighPassFilter,
        #     "HighPassFilter",
        #     "highpass_filter"
        # ),
        # (
        #     lowpass_filter.LowPassFilter,
        #     "LowPassFilter",
        #     "lowpass_filter"
        # ),
        # (
        #     flc.FLC,
        #     "FLC",
        #     "flc"
        # ),
        # (
        #     wflc.WFLC,
        #     "WFLC",
        #     "wflc"
        # ),
        # (
        #     bmflc.BMFLC,
        #     "BMFLC",
        #     "bmflc"
        # ),
        # (
        #     abpf.ABPF,
        #     "ABPF",
        #     "abpf"
        # ),
        # (
        #     bbf.BBF,
        #     "BBF",
        #     "bbf"
        # ),
        # (
        #     cdf.CDF,
        #     "CDF",
        #     "cdf"
        # ),
        # (
        #     kalman_filter.KF,
        #     "KF",
        #     "kalman_filter"
        # ),
        # (
        #     wflc_kf.WFLCKF,
        #     "WFLCKF",
        #     "wflc_kf"
        # ),
        # (
        #     bmflc_rls.BMFLC_RLS,
        #     "BMFLC_RLS",
        #     "bmflc_rls"
        # ),
        # (
        #     bmflc_kf.BMFLC_KF,
        #     "BMFLC_KF",
        #     "bmflc_kf"
        # ),
        # (
        #     ebmflc.EBMFLC,
        #     "EBMFLC",
        #     "ebmflc"
        # ),
        # (
        #     ar_lms.AR_LMS,
        #     "AR_LMS",
        #     "ar_lms"
        # ),
        # (
        #     ar_kf.AR_KF,
        #     "AR_KF",
        #     "ar_kf"
        # )
        # (
        #     as_bmflc.AS_BMFLC,
        #     "AS_BMFLC",
        #     "as_bmflc"
        # ),
        # (
        #     zpafkf.ZPAFKF,
        #     "ZPAFKF",
        #     "zpafkf"
        # ),
        # (
        #     amolc.AMOLC,
        #     "AMOLC",
        #     "amolc"
        # ),
        (
            ehwflc_kf.EHWFLC_KF,
            "EHWFLC_KF",
            "ehwflc_kf"
        ),
    ]

    # Demonstrate all methods in sequence
    for method_class, method_name, method_source in methods_to_demonstrate:
        demonstrate_method(method_class,
                           method_name,
                           input_signals,
                           method_source,
                           verbose=verbose)

    if verbose:
        logging.print_completion_message()


if __name__ == "__main__":
    main(verbose=False)
