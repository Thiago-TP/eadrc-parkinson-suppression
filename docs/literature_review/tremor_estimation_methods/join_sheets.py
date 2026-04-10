import argparse
from pathlib import Path

import pandas as pd


def stack_excel_files(
    file_stems: list[str],
    output_stem: str,
    output_dir: str,
    add_number_column: bool = False,
) -> Path:
    """Stack multiple .xlsx files into one, keeping only shared columns.

    Args:
        file_stems: List of file names without the .xlsx suffix.
        output_stem: Name of the output file without the .xlsx suffix.
        output_dir: Directory where the output file will be saved.
        add_number_column: Whether to create/overwrite a Number column.

    Returns:
        Path to the generated .xlsx file.
    """
    if not file_stems:
        raise ValueError("file_stems cannot be empty")

    input_paths = [Path(f"{stem}.xlsx") for stem in file_stems]

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

    dataframes: list = []
    common_columns: list[str] | None = None

    for path in input_paths:
        df = pd.read_excel(path)
        dataframes.append(df)

        if common_columns is None:
            common_columns = list(df.columns)
            continue

        current_columns = set(df.columns)
        common_columns = [
            col for col in common_columns if col in current_columns
        ]

    if common_columns is None or not common_columns:
        raise ValueError(
            "No common columns found across the provided Excel files"
        )

    filtered_frames = [df.loc[:, common_columns] for df in dataframes]
    merged_df = pd.concat(filtered_frames, ignore_index=True)

    if add_number_column:
        merged_df["Number"] = range(1, len(merged_df) + 1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{output_stem}.xlsx"

    merged_df.to_excel(output_path, index=False)
    return output_path


def resolve_input_stems(input_entries: list[str]) -> list[str]:
    """Resolve explicit names and wildcard patterns to file stems.

    Input values can be provided with or without the .xlsx suffix. If a value
    contains wildcard characters, all matching .xlsx files are expanded.
    """
    wildcard_chars = "*?[]"
    resolved_stems: list[str] = []
    seen: set[str] = set()

    for entry in input_entries:
        entry_path = Path(entry)
        no_suffix = (
            entry_path.with_suffix("")
            if entry_path.suffix == ".xlsx"
            else entry_path
        )
        stem_text = str(no_suffix)

        if any(char in stem_text for char in wildcard_chars):
            matches = sorted(Path(".").glob(f"{stem_text}.xlsx"))
            if not matches:
                raise FileNotFoundError(
                    f"No Excel files matched pattern: {entry}"
                )

            for match in matches:
                match_stem = str(match.with_suffix(""))
                if match_stem not in seen:
                    seen.add(match_stem)
                    resolved_stems.append(match_stem)
            continue

        if stem_text not in seen:
            seen.add(stem_text)
            resolved_stems.append(stem_text)

    return resolved_stems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stack multiple Excel files (.xlsx) into a single .xlsx file, "
            "keeping only columns common to all files."
        )
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output file name without the .xlsx extension.",
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help=(
            "Input file names/patterns with or without .xlsx "
            "(for example: databases_exports/file*)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the output file will be saved.",
    )
    parser.add_argument(
        "--add-number-column",
        action="store_true",
        help=(
            "Create or overwrite a 'Number' column in the output file with "
            "1-based row indices."
        ),
    )

    args = parser.parse_args()

    if not args.output_file:
        print("Warning: no output file was provided. Exiting.")
        return

    input_stems = resolve_input_stems(args.input_files)

    result = stack_excel_files(
        file_stems=input_stems,
        output_stem=args.output_file,
        output_dir=args.output_dir,
        add_number_column=args.add_number_column,
    )
    print(f"Created: {result}")


if __name__ == "__main__":
    main()
