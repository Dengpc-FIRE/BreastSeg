import argparse
import getpass
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import synapseclient


SYNAPSE_ID = "syn60868042"
DEFAULT_AUTH_TOKEN = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc4MTc1MzU2NiwiaWF0IjoxNzgxNzUzNTY2LCJqdGkiOiI0MDE1MSIsInN1YiI6IjM1NjI5MjUifQ.qo-BlnXhiVwBvn0YbmWKaPbufKt83gp-YNj0183XNVjMlx7AhpSatuwjEPrKdOLceWJWaZt3FvLQp4aZNGCXVNg-gUfUf8ecvcLo-lzJO3omS_kScdaBZ_K9rbb5zYLOg8O5uNSIozbe1-dmkmFGYWO3qRakrx9Y8sTVEiBU__KsEAKYaKks4tQv6ff957wes34WydY5D7NBk1x12IagNxN-PMnSq75Rc6yGtrKle2h_ZaQdJiiq5_9IaV-kpyDNkKob-XoFP1LhFEdgp4q-jHFJVpHZsnohP9ZfH3CrQJfdqtchgwWIEFlRQeV26-Edvq6oy5D6bV2IXagQBhrpTA"

COHORTS = ("DUKE", "ISPY1", "ISPY2", "NACT")
COMMON = "COMMON"
STATE_DIR_NAME = ".mama_mia_download_state"


@dataclass
class Counters:
    seen: int = 0
    complete: int = 0
    downloaded: int = 0
    bytes_seen: int = 0


@dataclass(frozen=True)
class RemoteFile:
    synapse_id: str
    relative_path: Path
    size: Optional[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream-download the MAMA-MIA dataset from Synapse by cohort."
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print less traversal progress.",
    )
    return parser.parse_args()


def normalize_cohort(value: str) -> str:
    value = value.upper().replace("-", "").replace("_", "")
    if value == "ALL":
        return "ALL"
    if value not in COHORTS:
        raise ValueError(f"Unknown cohort: {value}")
    return value


def selected_cohorts(values: Iterable[str]):
    normalized = [normalize_cohort(value) for value in values]
    if "ALL" in normalized:
        return set(COHORTS)
    return set(normalized)


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


def get_token() -> str:
    token = os.environ.get("SYNAPSE_AUTH_TOKEN") or DEFAULT_AUTH_TOKEN
    if not token:
        token = getpass.getpass("Synapse auth token: ")
    return token


def marker_path(output_dir: Path, cohort: str) -> Path:
    return output_dir / STATE_DIR_NAME / f"{cohort.lower()}.done"


def write_marker(output_dir: Path, cohort: str, counters: Counters) -> None:
    state_dir = output_dir / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    marker_path(output_dir, cohort).write_text(
        (
            f"cohort={cohort}\n"
            f"files={counters.seen}\n"
            f"bytes={counters.bytes_seen}\n"
            f"downloaded_this_run={counters.downloaded}\n"
        ),
        encoding="utf-8",
    )


def local_file_ok(output_dir: Path, remote_file: RemoteFile) -> bool:
    local_path = output_dir / remote_file.relative_path
    if not local_path.is_file():
        return False
    if remote_file.size is None:
        return True
    return local_path.stat().st_size == remote_file.size


def should_skip_bucket(
    bucket: str,
    selected: set,
    done_buckets: set,
    only_cohorts: bool,
) -> bool:
    if bucket == COMMON:
        return only_cohorts or bucket in done_buckets
    return bucket not in selected or bucket in done_buckets


def download_one(syn, output_dir: Path, bucket: str, remote_file: RemoteFile, overwrite: bool) -> bool:
    if local_file_ok(output_dir, remote_file) and not overwrite:
        return False

    local_path = output_dir / remote_file.relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    collision = "overwrite.local" if overwrite or local_path.exists() else "keep.local"
    print(f"[{bucket}] Downloading {remote_file.relative_path}", flush=True)
    syn.get(
        remote_file.synapse_id,
        downloadLocation=str(local_path.parent),
        ifcollision=collision,
    )
    return True


def print_summary(counters: Dict[str, Counters]) -> None:
    print("\nSummary:", flush=True)
    for bucket in (COMMON, *COHORTS):
        counter = counters[bucket]
        if counter.seen == 0:
            continue
        status = "done" if counter.seen == counter.complete else "missing"
        gb = counter.bytes_seen / (1024 ** 3)
        print(
            f"{bucket:>6}: {status:7} {counter.complete}/{counter.seen} files, "
            f"downloaded_this_run={counter.downloaded}, remote_size={gb:.3f} GB",
            flush=True,
        )


def walk_and_process(
    syn,
    parent_id: str,
    rel_dir: Path,
    inherited_cohort: Optional[str],
    output_dir: Path,
    selected: set,
    done_buckets: set,
    counters: Dict[str, Counters],
    args,
) -> None:
    shown_dir = str(rel_dir) if str(rel_dir) != "." else "."
    if not args.quiet:
        print(f"Listing {shown_dir}", flush=True)

    children = list(syn.getChildren(parent_id, includeTypes=["folder", "file"]))
    for child in children:
        name = child["name"]
        child_id = child["id"]
        rel_path = rel_dir / name
        child_cohort = inherited_cohort or cohort_from_name(name)

        if is_folder(child):
            bucket = child_cohort or COMMON
            if child_cohort and should_skip_bucket(bucket, selected, done_buckets, args.only_cohorts):
                if not args.quiet:
                    print(f"[{bucket}] Skipping folder {rel_path}", flush=True)
                continue
            walk_and_process(
                syn=syn,
                parent_id=child_id,
                rel_dir=rel_path,
                inherited_cohort=child_cohort,
                output_dir=output_dir,
                selected=selected,
                done_buckets=done_buckets,
                counters=counters,
                args=args,
            )
            continue

        if not is_file(child):
            continue

        bucket = child_cohort or COMMON
        if should_skip_bucket(bucket, selected, done_buckets, args.only_cohorts):
            continue

        remote_file = RemoteFile(
            synapse_id=child_id,
            relative_path=rel_path,
            size=child_size(child),
        )
        counter = counters[bucket]
        counter.seen += 1
        counter.bytes_seen += remote_file.size or 0

        already_ok = local_file_ok(output_dir, remote_file)
        if args.status:
            if already_ok:
                counter.complete += 1
            continue

        downloaded = False
        if not already_ok or args.overwrite:
            downloaded = download_one(syn, output_dir, bucket, remote_file, args.overwrite)
        if local_file_ok(output_dir, remote_file):
            counter.complete += 1
        if downloaded:
            counter.downloaded += 1


def main():
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_cohorts(args.cohorts)
    forced = selected_cohorts(args.force_cohort) if args.force_cohort else set()

    done_buckets = set()
    if not args.status:
        for bucket in (COMMON, *COHORTS):
            if bucket not in forced and marker_path(output_dir, bucket).is_file():
                done_buckets.add(bucket)

    if done_buckets:
        print(
            "Skipping buckets with .done markers: " + ", ".join(sorted(done_buckets)),
            flush=True,
        )

    syn = synapseclient.Synapse(skip_checks=True)
    syn.login(authToken=get_token())

    print(f"Streaming {SYNAPSE_ID} into {output_dir}", flush=True)
    counters = {bucket: Counters() for bucket in (COMMON, *COHORTS)}
    walk_and_process(
        syn=syn,
        parent_id=SYNAPSE_ID,
        rel_dir=Path(),
        inherited_cohort=None,
        output_dir=output_dir,
        selected=selected,
        done_buckets=done_buckets,
        counters=counters,
        args=args,
    )

    if not args.status:
        for bucket, counter in counters.items():
            if counter.seen > 0 and counter.seen == counter.complete:
                write_marker(output_dir, bucket, counter)
                print(f"[{bucket}] complete; wrote {marker_path(output_dir, bucket)}", flush=True)

    print_summary(counters)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
