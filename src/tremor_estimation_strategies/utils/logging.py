"""
Logging utilities for tremor estimation demonstration.

This module handles all logging/printing of results and progress messages.
"""


def print_header(text: str):
    """Print a formatted header."""
    print("=" * 70)
    print(text.upper())
    print("=" * 70)


def print_subheader(text: str):
    """Print a formatted subheader."""
    print("\n" + text)
    print("-" * 70)


def print_test_result(test_num: int | str, method, signal, estimates,
                      is_steady_state: bool = False, suffix: str = ""):
    """
    Print test results for a single test.

    Parameters:
    -----------
    test_num : int | str
        Test number (1, 2, or 3) or custom identifier
    method : object
        Method object with get_estimated_frequency() and freq_history
    signal : array-like
        Input signal array
    estimates : array-like
        Estimated signal array
    is_steady_state : bool
        Whether to show steady-state metrics
    suffix : str
        Additional text suffix for the test description
    """
    import numpy as np

    error_rms_full = np.sqrt(np.mean((signal - estimates) ** 2))
    error_rms_tail = np.sqrt(np.mean(
        (signal[int(len(signal)*0.5):] -
         estimates[int(len(signal)*0.5):]) ** 2
    ))

    # Determine test description
    test_descriptions = {
        1: "Fixed-Frequency Sinusoid (10 Hz)",
        2: "Frequency-Modulated Sinusoid",
    }
    description = test_descriptions.get(int(test_num), f"Test {test_num}")

    print_subheader(f"[TEST {test_num}] {description}{suffix}")

    print("True frequency: 10.00 Hz")

    # Show frequency estimates if available
    if hasattr(method, 'freq_history') and len(method.freq_history) > 0:
        print(f"Estimated frequency (initial): {method.freq_history[0]:.2f} Hz")  # noqa: E501

    print(f"Estimated frequency (final): {method.get_estimated_frequency():.2f} Hz")  # noqa: E501
    print(f"RMS Error (full): {error_rms_full:.4f}")
    print(f"RMS Error (steady-state, last 50%): {error_rms_tail:.4f}")
    print(f"Amplitude reduction: {(1 - error_rms_tail/np.std(signal)) * 100:.1f}%")  # noqa: E501


def print_start_message():
    """Print start message."""
    print("\n" + "=" * 70)
    print("TREMOR ESTIMATION METHODS - VOLUNTARY MOTION RECONSTRUCTION")
    print("=" * 70 + "\n")


def print_completion_message():
    """Print completion message."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("Generated signals saved in: input_examples/")
    print("  • periodic_signal.npz - Fixed-frequency 10 Hz tremor")
    print("  • modulated_signal.npz - Modulated")
    print("  • harmonic_signal.npz - Combined tremor (10 Hz) + voluntary motion (0.5 Hz)")  # noqa: E501
    print("=" * 70 + "\n")


def print_method_header(method_name: str):
    """Print method header."""
    print_header(f"{method_name} DEMONSTRATION")


def print_plot_saved(output_path: str):
    """Print plot saved confirmation."""
    print(f"\n* Plots saved to: {output_path}")
