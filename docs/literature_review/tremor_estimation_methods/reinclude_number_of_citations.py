import argparse
from pathlib import Path

import pandas as pd

_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"}


def _read_file(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame based on its extension."""
    if path.suffix.lower() in _EXCEL_SUFFIXES:
        return pd.read_excel(path)
    return pd.read_csv(path)


def insert_column_from_reference(
    reference_file: str | Path,
    target_files: list[str | Path],
    index_column: str,
    source_column: str,
    overwrite_existing: bool = True,
) -> list[Path]:
    """
    Insert a column from a reference file into one or more target Excel files.

    The reference file may be a CSV or an Excel file (.xlsx, .xls, etc.).
    Rows are matched between each target and the reference using index_column.
    The source_column values from the reference are merged into the target on
    that shared index key and the target file is overwritten in place.

    Parameters
    ----------
    reference_file
        Path (or name) of the reference file
        (.csv or .xlsx / other Excel formats).
    target_files
        List of paths (or names) of the target .xlsx files to be updated.
    index_column
        Column present in both the reference and every target file used to
        align rows.
    source_column
        Column from the reference file to be inserted into each target file.
    overwrite_existing
        If True and source_column already exists in a target, its values are
        replaced by those from the reference.  If False, existing columns are
        left untouched and only missing ones are added.

    Returns
    -------
    list[pathlib.Path]
        Paths of the updated target files.
    """
    reference_path = Path(reference_file)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    ref_df = _read_file(reference_path)

    if index_column not in ref_df.columns:
        raise ValueError(
            f"Index column '{index_column}' not found in reference file. "
            f"Available columns: {list(ref_df.columns)}"
        )
    if source_column not in ref_df.columns:
        raise ValueError(
            f"Source column '{source_column}' not found in reference file. "
            f"Available columns: {list(ref_df.columns)}"
        )

    lookup = ref_df[[index_column, source_column]].drop_duplicates(
        subset=index_column
    )

    updated_files: list[Path] = []

    for target in target_files:
        target_path = Path(target)
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")

        target_df = pd.read_excel(target_path)

        if index_column not in target_df.columns:
            raise ValueError(
                f"Index column '{index_column}' not found in target file "
                f"'{target_path}'. " +
                f"Available columns: {list(target_df.columns)}"
            )

        if source_column in target_df.columns and not overwrite_existing:
            updated_files.append(target_path)
            continue

        if source_column in target_df.columns:
            target_df = target_df.drop(columns=[source_column])

        merged = target_df.merge(lookup, on=index_column, how="left")

        merged.to_excel(target_path, index=False)
        updated_files.append(target_path)

    return updated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Insert a column from a reference CSV or Excel file "
            "into one or more target Excel files, "
            "matching rows by an index column."
        )
    )
    parser.add_argument(
        "reference_file",
        help="Path to the reference file (.csv or .xlsx).",
    )
    parser.add_argument(
        "target_files",
        nargs="+",
        help="One or more target .xlsx files to update.",
    )
    parser.add_argument(
        "--index-column",
        required=True,
        help="Column used to match rows between reference and targets.",
    )
    parser.add_argument(
        "--source-column",
        required=True,
        help="Column from the reference file to insert into each target.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip targets that already contain the source column.",
    )

    args = parser.parse_args()

    paths = insert_column_from_reference(
        reference_file=args.reference_file,
        target_files=args.target_files,
        index_column=args.index_column,
        source_column=args.source_column,
        overwrite_existing=not args.no_overwrite,
    )

    print("Updated files:")
    for p in paths:
        print(f" - {p}")
