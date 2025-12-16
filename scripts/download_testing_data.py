#!/usr/bin/env python3
"""
Download testing data using gdown and extract to data/ directory.

Usage:
    python scripts/download_testing_data.py
    python scripts/download_testing_data.py --file-id <google_drive_file_id>
    python scripts/download_testing_data.py --output-dir data/test
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================================
# Configuration: Fill in your Google Drive file ID here
# ============================================================================
# Get file ID from Google Drive share link:
# https://drive.google.com/file/d/FILE_ID_HERE/view
# Or from direct download link:
# https://drive.google.com/uc?id=FILE_ID_HERE
# ============================================================================
DEFAULT_FILE_ID = "1Hgxc-BOnI3dGPbiw_Tej3jYBIdbOV6Iw"  # TODO: Fill in your Google Drive file ID here
# ============================================================================


def check_gdown_installed() -> bool:
    """Check if gdown is installed."""
    try:
        import gdown
        return True
    except ImportError:
        return False


def install_gdown() -> None:
    """Install gdown if not available."""
    print("Installing gdown...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def download_file(url_or_id: str, output_path: Path, is_file_id: bool = False) -> Path:
    """
    Download file from Google Drive using gdown.
    
    Args:
        url_or_id: Google Drive URL or file ID
        output_path: Path to save the downloaded file
        is_file_id: If True, treat url_or_id as file ID instead of URL
    
    Returns:
        Path to downloaded file
    """
    import gdown
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_file_id:
        # Use file ID to construct URL
        url = f"https://drive.google.com/uc?id={url_or_id}"
    else:
        url = url_or_id
    
    print(f"Downloading from: {url}")
    print(f"Output path: {output_path}")
    
    try:
        gdown.download(url, str(output_path), quiet=False)
        print(f"✅ Downloaded successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """
    Extract archive file to target directory.
    
    Supports: .zip, .tar, .tar.gz, .tar.bz2, .tar.xz
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    suffix = archive_path.suffix.lower()
    
    print(f"Extracting {archive_path.name} to {extract_to}...")
    
    if suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        import tarfile
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.name.endswith(".tar.bz2") or archive_path.name.endswith(".tbz2"):
        import tarfile
        with tarfile.open(archive_path, "r:bz2") as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.name.endswith(".tar.xz"):
        import tarfile
        with tarfile.open(archive_path, "r:xz") as tar_ref:
            tar_ref.extractall(extract_to)
    elif suffix == ".tar":
        import tarfile
        with tarfile.open(archive_path, "r") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {suffix}")
    
    print(f"✅ Extracted successfully to: {extract_to}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download testing data using gdown and extract to data/ directory."
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Google Drive URL (e.g., https://drive.google.com/uc?id=FILE_ID or https://drive.google.com/file/d/FILE_ID/view)",
    )
    parser.add_argument(
        "--file-id",
        type=str,
        help="Google Drive file ID (alternative to --url)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for extracted files (default: data)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive file after extraction",
    )
    parser.add_argument(
        "--archive-name",
        type=str,
        help="Name for the downloaded archive file (default: auto-detect from URL)",
    )
    
    args = parser.parse_args()
    
    # Check if gdown is installed
    if not check_gdown_installed():
        print("⚠️  gdown not found. Installing...")
        try:
            install_gdown()
        except Exception as e:
            print(f"❌ Failed to install gdown: {e}")
            print("   Please install manually: pip install gdown")
            return 1
    
    # Determine file ID or URL
    file_id = args.file_id or DEFAULT_FILE_ID
    url = args.url
    
    # Validate arguments
    if not url and not file_id:
        parser.error(
            "Either --url or --file-id must be provided, "
            "or set DEFAULT_FILE_ID in the script"
        )
    
    if url and file_id:
        parser.error("Cannot specify both --url and --file-id")
    
    # Get script directory and repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    # Determine output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    
    # Determine archive file path
    if args.archive_name:
        archive_path = repo_root / args.archive_name
    else:
        # Auto-detect from URL or use default name
        if file_id:
            archive_name = f"testing_data_{file_id}.zip"
        elif url:
            # Try to extract filename from URL
            archive_name = "testing_data.zip"  # Default fallback
            if "/" in url:
                # Try to get filename from URL
                parts = url.split("/")
                for part in reversed(parts):
                    if part and "." in part:
                        archive_name = part.split("?")[0]  # Remove query params
                        break
        else:
            archive_name = "testing_data.zip"
        archive_path = repo_root / archive_name
    
    # Download file
    try:
        downloaded_file = download_file(
            url or file_id,
            archive_path,
            is_file_id=bool(file_id),
        )
    except Exception as e:
        print(f"❌ Failed to download file: {e}")
        return 1
    
    # Extract archive
    try:
        extract_archive(downloaded_file, output_dir)
    except Exception as e:
        print(f"❌ Failed to extract archive: {e}")
        return 1
    
    # Remove archive if not keeping it
    if not args.keep_archive:
        print(f"Removing archive file: {downloaded_file}")
        downloaded_file.unlink()
        print("✅ Archive removed")
    
    print(f"\n✅ Testing data downloaded and extracted successfully!")
    print(f"   Output directory: {output_dir}")
    print(f"   Contents: {list(output_dir.iterdir())}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

