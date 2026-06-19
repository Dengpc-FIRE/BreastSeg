import argparse
import getpass
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import synapseclient


SYNAPSE_ID = "syn60868042"
DEFAULT_AUTH_TOKEN = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc4MTc1MzU2NiwiaWF0IjoxNzgxNzUzNTY2LCJqdGkiOiI0MDE1MSIsInN1YiI6IjM1NjI5MjUifQ.qo-BlnXhiVwBvn0YbmWKaPbufKt83gp-YNj0183XNVjMlx7AhpSatuwjEPrKdOLceWJWaZt3FvLQp4aZNGCXVNg-gUfUf8ecvcLo-lzJO3omS_kScdaBZ_K9rbb5zYLOg8O5uNSIozbe1-dmkmFGYWO3qRakrx9Y8sTVEiBU__KsEAKYaKks4tQv6ff957wes34WydY5D7NBk1x12IagNxN-PMnSq75Rc6yGtrKle2h_ZaQdJiiq5_9IaV-kpyDNkKob-XoFP1LhFEdgp4q-jHFJVpHZsnohP9ZfH3CrQJfdqtchgwWIEFlRQeV26-Edvq6oy5D6bV2IXagQBhrpTA"

COHORTS = ("DUKE", "ISPY1", "ISPY2", "NACT")
STATE_DIR_NAME = ".mama_mia_download_state"


@dataclass(frozen=True)
class RemoteFile:
    synapse_id: str
    relative_path: Path
    size: Optional[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the MAMA-MIA dataset from Synapse by cohort."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Directory to download into. Default: current directory.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=list(COHORTS),
        help="Cohorts to download: DUKE ISPY1 ISPY2 NACT. Default: all.",
    )
    parser.add_argument(
        "--only-cohorts",
        action="store_true",
        help="Skip non-cohort shared files such as metadata and pretrained weights.",
    )
    parser.add_argument(
        "--force-cohort",
        action="append",
        default=[],
        help="Force re-check/re-download one cohort even if its .done marker exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local files instead of keeping verified existing files.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only print per-cohort completeness status; do not download.",
    )
    return parser.parse_args()


def normalize_cohort(value: str) -> str:
    value = value.upper().replace("-", "").replace("_", "")
    aliases = {
        "ALL": "ALL",
        "DUKE": "DUKE",
        "ISPY1": "ISPY1",
        "ISPY2": "ISPY2",
        "NACT": "NACT",
    }
    if value not in aliases:
        raise ValueError(f"Unknown cohort: {value}")
    return aliases[value]


def cohort_from_name(name: str) -> Optional[str]:
    lowered = name.lower()
    if lowered.startswith("duke_"):
        return "DUKE"
    if lowered.startswith("ispy1_"):
        return "ISPY1"
    if lowered.startswith("ispy2_"):
        return "ISPY2"
    if lowered.startswith("nact_"):
        return "NACT"
    return None


def child_type(child: dict) -> str:
    return str(child.get("type", "")).lower()


def is_folder(child: dict) -> bool:
    return "folder" in child_type(child)


def is_file(child: dict) -> bool:
    return "file" in child_type(child)


def child_size(child: dict) -> Optional[int]:
    for key in ("fileSize", "contentSize", "size"):
        value = child.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def selected_cohorts(values: Iterable[str]) -> List[str]:
    normalized = [normalize_cohort(value) for value in values]
    if "ALL" in normalized:
        return list(COHORTS)
    return [cohort for cohort in COHORTS if cohort in normalized]


def get_token() -> str:
    token = os.environ.get("SYNAPSE_AUTH_TOKEN") or DEFAULT_AUTH_TOKEN
    if not token:
        token = getpass.getpass("Synapse auth token: ")
    return token


def collect_remote_files(syn, root_id: str) -> Dict[str, List[RemoteFile]]:
    files: Dict[str, List[RemoteFile]] = {cohort: [] for cohort in COHORTS}
    files["COMMON"] = []

    def walk(parent_id: str, rel_dir: Path, inherited_cohort: Optional[str]) -> None:
        children = list(syn.getChildren(parent_id, includeTypes=["folder", "file"]))
        for child in children:
            name = child["name"]
            child_id = child["id"]
            rel_path = rel_dir / name
            cohort = inherited_cohort or cohort_from_name(name)

            if is_folder(child):
                walk(child_id, rel_path, cohort)
            elif is_file(child):
                bucket = cohort or "COMMON"
                files[bucket].append(
                    RemoteFile(
                        synapse_id=child_id,
                        relative_path=rel_path,
                        size=child_size(child),
                    )
                )

    walk(root_id, Path(), None)
    return files


def local_file_ok(output_dir: Path, remote_file: RemoteFile) -> bool:
    local_path = output_dir / remote_file.relative_path
    if not local_path.is_file():
        return False
    if remote_file.size is None:
        return True
    return local_path.stat().st_size == remote_file.size


def count_complete(output_dir: Path, remote_files: List[RemoteFile]) -> int:
    return sum(1 for remote_file in remote_files if local_file_ok(output_dir, remote_file))


def marker_path(output_dir: Path, cohort: str) -> Path:
    return output_dir / STATE_DIR_NAME / f"{cohort.lower()}.done"


def write_marker(output_dir: Path, cohort: str, remote_files: List[RemoteFile]) -> None:
    state_dir = output_dir / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    total_size = sum(remote_file.size or 0 for remote_file in remote_files)
    marker_path(output_dir, cohort).write_text(
        f"cohort={cohort}\nfiles={len(remote_files)}\nbytes={total_size}\n",
        encoding="utf-8",
    )


def cohort_is_done(output_dir: Path, cohort: str, remote_files: List[RemoteFile]) -> bool:
    if not remote_files:
        return False
    if not marker_path(output_dir, cohort).is_file():
        return False
    return count_complete(output_dir, remote_files) == len(remote_files)


def print_status(output_dir: Path, name: str, remote_files: List[RemoteFile]) -> None:
    complete = count_complete(output_dir, remote_files)
    total = len(remote_files)
    total_size = sum(remote_file.size or 0 for remote_file in remote_files)
    status = "done" if total > 0 and complete == total else "missing"
    print(
        f"{name:>6}: {status:7} {complete}/{total} files, "
        f"remote_size={total_size / (1024 ** 3):.3f} GB"
    )


def download_files(syn, output_dir: Path, name: str, remote_files: List[RemoteFile], overwrite: bool) -> None:
    total = len(remote_files)
    for index, remote_file in enumerate(remote_files, start=1):
        local_path = output_dir / remote_file.relative_path
        if not overwrite and local_file_ok(output_dir, remote_file):
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{name} {index}/{total}] Downloading {remote_file.relative_path}")
        syn.get(
            remote_file.synapse_id,
            downloadLocation=str(local_path.parent),
            ifcollision="overwrite.local" if overwrite else "keep.local",
        )


def main():
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cohorts = selected_cohorts(args.cohorts)
    forced = set(selected_cohorts(args.force_cohort)) if args.force_cohort else set()

    syn = synapseclient.Synapse(skip_checks=True)
    syn.login(authToken=get_token())

    print(f"Collecting remote file list for {SYNAPSE_ID}...")
    remote = collect_remote_files(syn, SYNAPSE_ID)

    if args.status:
        if not args.only_cohorts:
            print_status(output_dir, "COMMON", remote["COMMON"])
        for cohort in cohorts:
            print_status(output_dir, cohort, remote[cohort])
        return

    if not args.only_cohorts:
        common = remote["COMMON"]
        if cohort_is_done(output_dir, "COMMON", common):
            print("[COMMON] already complete; skipping.")
        else:
            print_status(output_dir, "COMMON", common)
            download_files(syn, output_dir, "COMMON", common, args.overwrite)
            if count_complete(output_dir, common) == len(common):
                write_marker(output_dir, "COMMON", common)
                print("[COMMON] complete.")

    for cohort in cohorts:
        cohort_files = remote[cohort]
        if cohort not in forced and cohort_is_done(output_dir, cohort, cohort_files):
            print(f"[{cohort}] already complete; skipping whole cohort.")
            continue

        print_status(output_dir, cohort, cohort_files)
        download_files(syn, output_dir, cohort, cohort_files, args.overwrite)
        if count_complete(output_dir, cohort_files) == len(cohort_files):
            write_marker(output_dir, cohort, cohort_files)
            print(f"[{cohort}] complete.")
        else:
            complete = count_complete(output_dir, cohort_files)
            print(f"[{cohort}] incomplete: {complete}/{len(cohort_files)} files present.")

    print(f"Done checking/downloading cohorts {', '.join(cohorts)} to {output_dir}")


if __name__ == "__main__":
    main()
