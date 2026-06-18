import argparse
import getpass
import os
from pathlib import Path

import synapseclient
import synapseutils


SYNAPSE_ID = "syn60868042"
DEFAULT_AUTH_TOKEN = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc4MTc1MzU2NiwiaWF0IjoxNzgxNzUzNTY2LCJqdGkiOiI0MDE1MSIsInN1YiI6IjM1NjI5MjUifQ.qo-BlnXhiVwBvn0YbmWKaPbufKt83gp-YNj0183XNVjMlx7AhpSatuwjEPrKdOLceWJWaZt3FvLQp4aZNGCXVNg-gUfUf8ecvcLo-lzJO3omS_kScdaBZ_K9rbb5zYLOg8O5uNSIozbe1-dmkmFGYWO3qRakrx9Y8sTVEiBU__KsEAKYaKks4tQv6ff957wes34WydY5D7NBk1x12IagNxN-PMnSq75Rc6yGtrKle2h_ZaQdJiiq5_9IaV-kpyDNkKob-XoFP1LhFEdgp4q-jHFJVpHZsnohP9ZfH3CrQJfdqtchgwWIEFlRQeV26-Edvq6oy5D6bV2IXagQBhrpTA"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the MAMA-MIA dataset from Synapse."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Directory to download into. Default: current directory.",
    )
    parser.add_argument(
        "--manifest",
        default="all",
        choices=["all", "root", "suppress"],
        help="Manifest behavior. Default: all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local files instead of keeping existing files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("SYNAPSE_AUTH_TOKEN") or DEFAULT_AUTH_TOKEN
    if not token:
        token = getpass.getpass("Synapse auth token: ")

    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    collision = "overwrite.local" if args.overwrite else "keep.local"
    synapseutils.syncFromSynapse(
        syn=syn,
        entity=SYNAPSE_ID,
        path=str(output_dir),
        ifcollision=collision,
        manifest=args.manifest,
        downloadFile=True,
    )

    print(f"Done downloading {SYNAPSE_ID} to {output_dir}")


if __name__ == "__main__":
    main()
