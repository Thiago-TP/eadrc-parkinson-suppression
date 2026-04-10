import numpy as np
import pandas as pd
from pathlib import Path


def table_results(test_case: str = "Open loop response"):
    """
    Compare RMSE of voluntary motion estimates across all estimation methods.

    Args:
        test_case: The test case key to compare (default: "Open loop response")
                   Can be one of:
                   "Fixed Frequency Tremor",
                   "Modulated Frequency Tremor",
                   "Open loop response"

    Returns:
        pd.DataFrame: DataFrame with columns ["Method", "RMSE"] sorted by RMSE
    """
    results_path = Path("tremor_estimation/results")
    results_list = []

    # Iterate through all subdirectories in results
    for method_dir in sorted(results_path.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name

        # Find the NPZ file in this method directory
        npz_files = list(method_dir.glob("*.npz"))
        if not npz_files:
            continue

        npz_file = npz_files[0]

        try:
            # Load the NPZ file
            data = np.load(npz_file, allow_pickle=True)

            # Extract the test case data
            if test_case not in data:
                print(f"Warning: {test_case} not found in {method_name}")
                continue

            # Convert numpy array to dictionary
            test_data = data[test_case].item()

            # Extract volunteer motion arrays
            true_voluntary = test_data["true_voluntary"]
            voluntary_estimates = test_data["voluntary_estimates"]

            # Calculate RMSE
            rmse = np.sqrt(
                np.mean((voluntary_estimates - true_voluntary) ** 2))

            results_list.append(
                {"Method": method_name, "Voluntary Motion RMSE": rmse}
            )

        except Exception as e:
            print(f"Error processing {method_name}: {e}")
            continue

    # Create DataFrame and sort by RMSE
    df = pd.DataFrame(results_list)
    df = df.sort_values("Voluntary Motion RMSE").reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Test the function
    results_df = table_results()
    print(results_df)
