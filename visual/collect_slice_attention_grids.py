from __future__ import annotations

import argparse
import shutil
from pathlib import Path


GRID_PATTERNS = {
    "effective_slice_grid": "*effective_slice_grid*.png",
    "raw_slice_grid": "*raw_slice_grid*.png",
}


def default_source() -> Path:
    candidates = [
        Path("results_kpta_25d_net") / "visual" / "test" / "slice_attention",
        Path("visual") / "test" / "slice_attention",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect only CSAM effective/raw slice grid images from a "
            "slice_attention visualization directory."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source(),
        help=(
            "Input slice_attention directory. Default: "
            "results_kpta_25d_net/visual/test/slice_attention if it exists, "
            "otherwise visual/test/slice_attention."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <source_parent>/slice_attention_selected_grids."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print files that would be copied without copying them.",
    )
    return parser.parse_args()


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def collect_grids(source: Path, output: Path, overwrite: bool, dry_run: bool) -> dict[str, int]:
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    counts: dict[str, int] = {}
    output.mkdir(parents=True, exist_ok=True)

    for category, pattern in GRID_PATTERNS.items():
        category_dir = output / category
        if not dry_run:
            category_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(path for path in source.rglob(pattern) if path.is_file())
        counts[category] = len(files)

        for src in files:
            relative_parent = src.relative_to(source).parent
            dst_dir = category_dir / relative_parent
            dst = dst_dir / src.name
            if not overwrite:
                dst = unique_destination(dst)
            if dry_run:
                print(f"[dry-run] {src} -> {dst}")
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"copied {src} -> {dst}")

    return counts


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else (source.parent / "slice_attention_selected_grids").resolve()
    )
    counts = collect_grids(source, output, args.overwrite, args.dry_run)
    total = sum(counts.values())
    print(f"source: {source}")
    print(f"output: {output}")
    print(f"effective_slice_grid: {counts.get('effective_slice_grid', 0)}")
    print(f"raw_slice_grid: {counts.get('raw_slice_grid', 0)}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
