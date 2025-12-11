#!/usr/bin/env python3
"""Download and extract the GeoBench dataset from Hugging Face."""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

DATASET_REPO = "recursix/geo-bench-1.0"


def decompress_zip_with_progress(
    zip_file_path: Path, extract_to_folder: Path | None = None
) -> None:
    """Decompress a zip file with a progress bar and remove the zip file.

    Args:
        zip_file_path: Path to the zip file to decompress.
        extract_to_folder: Directory to extract files to. Defaults to zip file's parent directory.
    """
    extract_to_folder = extract_to_folder or zip_file_path.parent

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_names = zip_ref.namelist()

        with tqdm(
            total=len(file_names), unit="file", desc=f"Extracting {zip_file_path.name}"
        ) as pbar:
            for file in file_names:
                zip_ref.extract(file, extract_to_folder)
                pbar.update(1)

    zip_file_path.unlink()
    print(f"Removed zip file: {zip_file_path}")


def download_benchmark(local_directory: Path | str, force: bool = False) -> None:
    """Download and extract the GeoBench dataset from Hugging Face.

    Args:
        local_directory: Directory to download the dataset to.
            Defaults to GEO_BENCH_DIR environment variable or "data/".
        force: Force re-download of files even if they already exist.
    """
    local_directory = Path(local_directory)

    local_directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GeoBench dataset to: {local_directory}")

    api = HfApi()
    dataset_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")

    _download_files(dataset_files, local_directory, force)
    _decompress_files(dataset_files, local_directory, force)

    print("Download and decompression process completed.")


def _download_files(dataset_files: list[str], local_directory: Path, force: bool = False) -> None:
    """Download all files from the dataset repository.

    Args:
        dataset_files: List of files to download.
        local_directory: Directory to download files to.
        force: Force re-download of files even if they already exist.
    """
    for file in dataset_files:
        local_file_path = local_directory / file

        # Skip if file already exists (unless force is enabled)
        if local_file_path.exists() and not force:
            print(f"Skipping {file} (already exists)")
            continue

        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {file}...")
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=file,
            cache_dir=local_directory,
            local_dir=local_directory,
            repo_type="dataset",
        )


def _decompress_files(dataset_files: list[str], local_directory: Path, force: bool = False) -> None:
    """Decompress all zip files from the dataset.

    Args:
        dataset_files: List of all dataset files.
        local_directory: Directory containing zip files.
        force: Force re-extraction even if already extracted.
    """
    zip_files = [file for file in dataset_files if file.endswith(".zip")]

    for i, zip_file in enumerate(zip_files, start=1):
        zip_file_path = local_directory / zip_file

        # Skip if zip file doesn't exist (already extracted and removed) and not forcing
        if not zip_file_path.exists() and not force:
            print(f"Skipping {zip_file} (already extracted)")
            continue

        print(f"Decompressing {i}/{len(zip_files)}: {zip_file}...")
        decompress_zip_with_progress(zip_file_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract the GeoBench dataset from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/",
        help="Directory to download and extract the dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of files even if they already exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.force:
        print("Force mode enabled: existing files will be re-downloaded")

    download_benchmark(args.output_dir, args.force)
