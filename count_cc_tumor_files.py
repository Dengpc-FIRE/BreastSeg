from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path("CC-Tumor-Heterogeneity")


@dataclass(frozen=True)
class CountRow:
    patient: str
    total_files: int
    direct_files: int
    subdirs: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count files under each CCTH patient directory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="CC-Tumor-Heterogeneity dataset root.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Also print per-immediate-subdirectory recursive file counts.",
    )
    return parser.parse_args()


def find_patient_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(
        (path for path in root.rglob("CCTH-*") if path.is_dir()),
        key=lambda path: path.name,
    )


def count_files_recursive(path: Path) -> int:
    return sum(1 for child in path.rglob("*") if child.is_file())


def count_patient(path: Path) -> CountRow:
    direct_children = list(path.iterdir())
    return CountRow(
        patient=path.name,
        total_files=count_files_recursive(path),
        direct_files=sum(1 for child in direct_children if child.is_file()),
        subdirs=sum(1 for child in direct_children if child.is_dir()),
        path=path,
    )


def print_table(rows: list[CountRow]) -> None:
    headers = ("patient", "total_files", "direct_files", "subdirs", "path")
    table = [
        (
            row.patient,
            str(row.total_files),
            str(row.direct_files),
            str(row.subdirs),
            str(row.path),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print(f"\npatients={len(rows)} total_files={sum(row.total_files for row in rows)}")


def print_detail(patient_dirs: list[Path]) -> None:
    for patient_dir in patient_dirs:
        child_dirs = sorted(
            (child for child in patient_dir.iterdir() if child.is_dir()),
            key=lambda path: path.name,
        )
        print(f"\n[{patient_dir.name}]")
        if not child_dirs:
            print("  no immediate subdirectories")
            continue
        for child_dir in child_dirs:
            print(f"  {child_dir.name}: {count_files_recursive(child_dir)}")


def write_csv(rows: list[CountRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patient", "total_files", "direct_files", "subdirs", "path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "patient": row.patient,
                    "total_files": row.total_files,
                    "direct_files": row.direct_files,
                    "subdirs": row.subdirs,
                    "path": row.path,
                }
            )


def main() -> int:
    args = parse_args()
    patient_dirs = find_patient_dirs(args.root)
    if not patient_dirs:
        raise RuntimeError(f"No CCTH-* patient directories found under: {args.root}")
    rows = [count_patient(path) for path in patient_dirs]
    print_table(rows)
    if args.detail:
        print_detail(patient_dirs)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nwrote_csv={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
