"""
Visualization module for tremor estimation results.

Provides functions to plot and save results from tremor estimation demonstrations.
"""

import os
import matplotlib.pyplot as plt


def _plot_reference_vs_estimate(ax,
                                time,
                                reference,
                                estimate,
                                method_name,
                                reference_label,
                                case_study):
    """Plot reference vs estimate with optional true component."""
    ax.plot(time,
            reference,
            color='tab:blue',
            linestyle='-',
            label=reference_label,
            linewidth=1.5)
    ax.plot(time,
            estimate,
            color='tab:red',
            linestyle='--',
            label=f'{method_name} estimate',
            linewidth=1.5)
    ax.plot(time,
            reference - estimate,
            color='tab:orange',
            linestyle=':',
            label='Error',
            linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(case_study)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def _plot_test_column(axes_col,
                      data_dict,
                      method_name,
                      case_study,
                      t_slice=1.0):
    """Encapsulate plotting logic for a single test col."""

    s = slice(0, int(t_slice * len(data_dict["t"])))  # cut signals for clarity

    t = data_dict["t"][s]
    signal = data_dict["signal"][s]
    true_tremor = data_dict["true_tremor"][s]
    tremor_estimates = data_dict["tremor_estimates"][s]
    true_voluntary = data_dict["true_voluntary"][s]
    voluntary_estimates = data_dict["voluntary_estimates"][s]

    # Top row: input signal vs its reconstruction
    ax_top = axes_col[0]
    _plot_reference_vs_estimate(
        ax_top, t, signal, voluntary_estimates + tremor_estimates, method_name,
        reference_label="Input signal",
        case_study=case_study
    )

    # Middle row: true tremor and its reconstruction
    ax_middle = axes_col[1]
    _plot_reference_vs_estimate(
        ax_middle, t, true_tremor, tremor_estimates, method_name,
        reference_label="True tremor",
        case_study="",
    )

    # Bottom row: true voluntary and its reconstruction
    ax_bottom = axes_col[2]
    _plot_reference_vs_estimate(
        ax_bottom, t, true_voluntary, voluntary_estimates, method_name,
        reference_label="True voluntary motion",
        case_study="",
    )


def plot_demonstration_results(results: dict,
                               method_name: str,
                               method_source: str,
                               output_base: str = "results") -> None:
    """
    Generate and save comprehensive visualization of tremor estimation results.

    Creates a comprehensive subplot figure showing for each test:
    - Signal vs estimate (and true component if available)
    - Voluntary motion reconstruction

    Parameters:
    -----------
    results : dict
        Dictionary containing results from demonstrate_method() with keys:
        "test1", "test2", each containing:
        - "t": time vector
        - "signal": input signal
        - "tremor_estimates": method estimates (tremor for test 3)
        - "true_voluntary": true voluntary motion component
        - "voluntary_estimates": estimated/reconstructed voluntary motion
    method_name : str
        Display name of the method
    method_source : str
        Source module name without extension (e.g., "wflc" for methods/wflc.py)
    output_base : str
        Base directory for results

    Returns:
    --------
    None
    """

    # Each test is a column with 3 rows:
    # 1) input signal vs input estimate
    # 2) true tremor vs tremor estimate
    # 3) true voluntary vs voluntary estimate
    n_rows = 3
    n_tests = len(results)
    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=n_tests,
                             figsize=(10 * n_tests, 4 * n_rows),
                             sharex='col',
                             sharey='col')

    # Plot each column
    for col, (case_study, data_dict) in enumerate(results.items()):
        _plot_test_column(axes[:, col] if n_tests > 1 else axes,
                          data_dict,
                          method_name,
                          case_study)

    # Create output directory and file
    output_dir = os.path.join(output_base, method_source)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{method_source}_demonstration_results.pdf"
    output_path = os.path.join(output_dir, filename)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"\t* Plots saved to: {output_path}")

    return
