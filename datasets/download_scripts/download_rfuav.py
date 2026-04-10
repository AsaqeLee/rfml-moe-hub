#!/usr/bin/env python3
"""
Download RFUAV dataset from HuggingFace.

RFUAV: 37 drone/RC types, 102 GB compressed (.rar), ~263 GB extracted.
Source: https://huggingface.co/datasets/kitofrank/RFUAV

Requirements:
    pip install huggingface-hub
    sudo apt install unrar  # or: sudo pacman -S unrar

Usage:
    python download_rfuav.py [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Download RFUAV dataset")
    parser.add_argument("--output-dir", default="data/rfuav_compressed",
                        help="Directory to download compressed files")
    parser.add_argument("--extract-dir", default="data/rfuav_raw",
                        help="Directory to extract IQ files")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only extract")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction, only download")
    args = parser.parse_args()

    if not args.skip_download:
        print("Downloading RFUAV dataset from HuggingFace...")
        print("This will download ~102 GB of compressed .rar files.")
        print(f"Output directory: {args.output_dir}")

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
            sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)
        snapshot_download(
            "kitofrank/RFUAV",
            local_dir=args.output_dir,
            repo_type="dataset",
            resume_download=True
        )
        print(f"Download complete: {args.output_dir}")

    if not args.skip_extract:
        print(f"\nExtracting .rar files to {args.extract_dir}...")
        os.makedirs(args.extract_dir, exist_ok=True)

        rar_files = sorted(f for f in os.listdir(args.output_dir) if f.endswith('.rar'))
        print(f"Found {len(rar_files)} .rar files to extract")

        for i, rar_file in enumerate(rar_files, 1):
            rar_path = os.path.join(args.output_dir, rar_file)
            print(f"[{i}/{len(rar_files)}] Extracting {rar_file}...")
            result = subprocess.run(
                ["unrar", "x", "-o-", rar_path, args.extract_dir],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  WARNING: Failed to extract {rar_file}: {result.stderr[:200]}")
            else:
                print(f"  OK: {rar_file}")

        print(f"\nExtraction complete: {args.extract_dir}")

    # Verify
    if os.path.isdir(args.extract_dir):
        drone_dirs = [d for d in os.listdir(args.extract_dir)
                      if os.path.isdir(os.path.join(args.extract_dir, d))]
        total_iq = 0
        for d in drone_dirs:
            for root, dirs, files in os.walk(os.path.join(args.extract_dir, d)):
                total_iq += sum(1 for f in files if f.endswith('.iq'))
        print(f"\nVerification: {len(drone_dirs)} drone directories, {total_iq} .iq files")


if __name__ == "__main__":
    main()
