import argparse
import re
from pathlib import Path

import pandas as pd


def _safe_file_fragment(value: object) -> str:
    """Convert a category value to a safe file-name fragment."""
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]", "", text)
    return text or "EMPTY"


def split_excel_by_category(
        excel_name_no_ext: str,
        category_column: str,
        input_dir: str | Path = ".",
        output_dir: str | Path | None = None,
        na_label: str = "MISSING",
) -> list[Path]:
    """Split an Excel file into multiple files by one categorical column.

    Parameters
    ----------
    excel_name_no_ext
            Input Excel file name without the .xlsx extension.
    category_column
            Column used to split the data.
    input_dir
            Directory containing the input file.
    output_dir
            Directory where output files are written.
            If None, files are written in input_dir.
    na_label
            Label used when category values are missing (NaN).

    Returns
    -------
    list[pathlib.Path]
            Paths of generated Excel files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{excel_name_no_ext}.xlsx"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path)
    if category_column not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Column '{category_column}' not found. "
            f"Available columns: {available}"
        )

    written_files: list[Path] = []

    # Keep missing values as their own group so they can also be exported.
    grouped_df = df.groupby(df[category_column].fillna(na_label), dropna=False)
    for value, group in grouped_df:
        safe_value = _safe_file_fragment(value)
        out_name = f"{excel_name_no_ext}_{safe_value}.xlsx"
        out_path = output_dir / out_name
        group.to_excel(out_path, index=False)
        written_files.append(out_path)

    return written_files


if __name__ == "__main__":
    # Example:
    # python adherance_checker.py input_file_name ADHERANCE ./stacked_excels

    parser = argparse.ArgumentParser(
        description="Split one Excel file into multiple files by category."
    )
    parser.add_argument(
        "excel_name_no_ext",
        help="Excel file name without .xlsx",
    )
    parser.add_argument(
        "category_column",
        help="Name of the categorical column used to split",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Directory where split files are written",
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing the input Excel file",
    )
    parser.add_argument(
        "--na-label",
        default="MISSING",
        help="Label used to name file for missing values",
    )

    args = parser.parse_args()
    files = split_excel_by_category(
        excel_name_no_ext=args.excel_name_no_ext,
        category_column=args.category_column,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        na_label=args.na_label,
    )

    print("Created files:")
    for path in files:
        print(f"- {path}")
