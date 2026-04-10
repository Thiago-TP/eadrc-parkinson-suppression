from __future__ import annotations

import argparse
import csv
from pathlib import Path

MAX_LINES_PER_FILE = 51  # n works plus 1 for the header


def split_csv_file(
    file_stem: str,
    max_lines: int = MAX_LINES_PER_FILE,
    output_dir: str = "csvs"
) -> list[Path]:
    if max_lines < 2:
        raise ValueError("max_lines must be at least 2 to preserve the header")

    source_path = Path(f"{file_stem}.csv")

    if not source_path.exists():
        raise FileNotFoundError(f"CSV file not found: {source_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    file_index = 1
    current_line_count = 0
    header: list[str] | None = None
    output_file = None
    writer = None

    with source_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = next(reader, None)

        if header is None:
            return created_files

        for row in reader:
            if writer is None or current_line_count >= max_lines:
                if output_file is not None:
                    output_file.close()

                output_path = out_dir / \
                    f"{source_path.stem}_part{file_index}.csv"
                output_file = output_path.open(
                    "w", newline="", encoding="utf-8"
                )
                writer = csv.writer(output_file)
                writer.writerow(header)
                created_files.append(output_path)
                file_index += 1
                current_line_count = 1

            writer.writerow(row)
            current_line_count += 1

        if writer is None:
            output_path = out_dir / f"{source_path.stem}_part{file_index}.csv"
            output_file = output_path.open(
                "w", newline="", encoding="utf-8"
            )
            writer = csv.writer(output_file)
            writer.writerow(header)
            created_files.append(output_path)

    if output_file is not None:
        output_file.close()

    return created_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a CSV file into smaller CSV files with at most "
            "50 lines each, preserving the header in every part."
        )
    )
    parser.add_argument(
        "file_name",
        help="Name of the CSV file without the .csv extension.",
    )
    args = parser.parse_args()

    created_files = split_csv_file(args.file_name)

    if not created_files:
        print("No output files were created.")
        return

    print(f"Created {len(created_files)} file(s):")
    for path in created_files:
        print(path)


if __name__ == "__main__":
    main()
