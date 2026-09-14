"""Download and verify the MatchLens dataset release."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://github.com/hugopicot/fotchebal-so-ney-do-brazzzzil/"
    "releases/download/dataset-v1.0.0/dataset.zip"
)
REPOSITORY = "hugopicot/fotchebal-so-ney-do-brazzzzil"
RELEASE_TAG = "dataset-v1.0.0"
DATASET_SHA256 = "6827840b53574f25ce135522c8907a9f19f494596126d8e1be32d6db8068ac6e"
EXPECTED_FILES = {
    "sample_submission.csv",
    "test.csv",
    "train.csv",
    "train_target_and_scores.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, verify and extract the MatchLens CSV dataset."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.cwd(),
        help="Directory receiving the CSV files (default: current directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace CSV files that already exist in the destination.",
    )
    return parser.parse_args()


def download_archive(archive: Path) -> None:
    print(f"Downloading {DATASET_URL}")
    try:
        with urllib.request.urlopen(DATASET_URL) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        return
    except urllib.error.HTTPError as error:
        if error.code not in {403, 404}:
            raise

    if shutil.which("gh") is None:
        raise SystemExit(
            "The release is private. Install GitHub CLI and run 'gh auth login', "
            "or download dataset.zip manually from the release."
        )

    print("Public download unavailable; using the authenticated GitHub CLI.")
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            RELEASE_TAG,
            "--repo",
            REPOSITORY,
            "--pattern",
            "dataset.zip",
            "--dir",
            str(archive.parent),
            "--clobber",
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    existing = sorted(name for name in EXPECTED_FILES if (destination / name).exists())
    if existing and not args.force:
        names = ", ".join(existing)
        raise SystemExit(f"Existing dataset files: {names}. Use --force to replace them.")

    with tempfile.TemporaryDirectory(prefix="matchlens-dataset-") as temp_dir:
        archive = Path(temp_dir) / "dataset.zip"
        download_archive(archive)

        actual_hash = sha256(archive)
        if actual_hash != DATASET_SHA256:
            raise SystemExit(
                f"Checksum mismatch: expected {DATASET_SHA256}, got {actual_hash}"
            )

        with zipfile.ZipFile(archive) as dataset_zip:
            members = {name for name in dataset_zip.namelist() if not name.endswith("/")}
            if members != EXPECTED_FILES:
                raise SystemExit(
                    f"Unexpected archive content: {', '.join(sorted(members))}"
                )
            dataset_zip.extractall(destination)

    print(f"Dataset extracted to {destination}")


if __name__ == "__main__":
    main()
