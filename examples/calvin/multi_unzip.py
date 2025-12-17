#!/usr/bin/env python3
"""
High-performance resume-capable ZIP extractor for network drives.
Optimized for many small files with threading (I/O-bound workload).
"""

import zipfile
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
import threading

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Install it with: pip install tqdm")
    exit(1)


class ZipExtractor:
    def __init__(self, zip_path: str, output_dir: str, overwrite: bool = False):
        self.zip_path = Path(zip_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.overwrite = overwrite

        # Thread-local storage for ZipFile handles
        self._local = threading.local()
        self._lock = Lock()

        # Statistics
        self.stats = {
            "extracted": 0,
            "skipped": 0,
            "errors": 0,
        }
        self.stats_lock = Lock()

    def _get_zip_handle(self):
        """Get a thread-local ZipFile handle."""
        if not hasattr(self._local, "zf"):
            self._local.zf = zipfile.ZipFile(self.zip_path, "r")
        return self._local.zf

    def extract_file(self, member_info: tuple) -> tuple:
        """
        Extract a single file. Returns (status, filename, size).
        """
        filename, file_size, is_dir = member_info

        try:
            dest_path = self.output_dir / filename

            # Skip directories
            if is_dir:
                dest_path.mkdir(parents=True, exist_ok=True)
                return ("dir", filename, 0)

            # Check if already exists with correct size
            if not self.overwrite and dest_path.exists():
                try:
                    if dest_path.stat().st_size == file_size:
                        return ("skipped", filename, file_size)
                except OSError:
                    pass  # File might have been deleted, extract it

            # Create parent directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract using thread-local ZipFile handle
            zf = self._get_zip_handle()
            data = zf.read(filename)

            # Write to file
            with open(dest_path, "wb") as f:
                f.write(data)

            return ("extracted", filename, file_size)

        except Exception as e:
            return ("error", filename, 0, str(e))

    def update_stats(self, status: str, size: int):
        with self.stats_lock:
            if status == "extracted":
                self.stats["extracted"] += 1
            elif status == "skipped":
                self.stats["skipped"] += 1
            elif status == "error":
                self.stats["errors"] += 1


def extract_parallel(
    zip_path: str, output_dir: str, overwrite: bool = False, num_workers: int = 32, batch_size: int = 1000
):
    """
    Extract ZIP file using thread pool.

    For network drives with many small files, use high thread count (32-64).
    """
    zip_path = Path(zip_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get file list from ZIP
    print("Reading ZIP file index...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [(m.filename, m.file_size, m.is_dir()) for m in zf.infolist()]

    total_files = len(members)
    total_size = sum(m[1] for m in members)

    print(f"Total files: {total_files:,}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Workers: {num_workers}")
    print("-" * 60)

    extractor = ZipExtractor(zip_path, output_dir, overwrite)
    errors = []

    with tqdm(total=total_files, unit="files", desc="Extracting", smoothing=0.1, mininterval=0.5) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit in batches to avoid memory issues with 1.9M files
            for batch_start in range(0, total_files, batch_size):
                batch_end = min(batch_start + batch_size, total_files)
                batch = members[batch_start:batch_end]

                futures = {executor.submit(extractor.extract_file, m): m for m in batch}

                for future in as_completed(futures):
                    result = future.result()
                    status = result[0]

                    extractor.update_stats(status, result[2])

                    if status == "error":
                        errors.append((result[1], result[3]))

                    pbar.update(1)
                    pbar.set_postfix(
                        extracted=extractor.stats["extracted"],
                        skipped=extractor.stats["skipped"],
                        errors=extractor.stats["errors"],
                        refresh=False,
                    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETE!")
    print(f"  Extracted: {extractor.stats['extracted']:,} files")
    print(f"  Skipped:   {extractor.stats['skipped']:,} files")
    print(f"  Errors:    {extractor.stats['errors']:,} files")
    print(f"{'=' * 60}")

    if errors:
        print(f"\nFirst 10 errors:")
        for fn, err in errors[:10]:
            print(f"  {fn}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Fast parallel ZIP extractor for network drives")
    parser.add_argument("zip_file", help="Path to the ZIP file")
    parser.add_argument("-o", "--output", default=".", help="Output directory (default: current directory)")
    parser.add_argument(
        "-w", "--workers", type=int, default=32, help="Number of threads (default: 32, try 64-128 for network drives)"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1000, help="Batch size for submitting jobs (default: 1000)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    try:
        extract_parallel(args.zip_file, args.output, args.overwrite, args.workers, args.batch_size)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Run again to resume.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
