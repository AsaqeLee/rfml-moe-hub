#!/usr/bin/env python3
"""
Download DRFF-R2 dataset from SciDB China.

DRFF-R2: Multi-scenario UAV RF dataset, 400 GB, 730 files across 7 scenarios.
Source: china.scidb.cn, dataSetId=b8a16448c1284fd1be1ded9ccc45be20

The dataset is organized as:
  V3/dataset1/ - Single drone states
  V3/dataset2/ - Drone mixed
  V3/dataset3/ - Hover
  V3/dataset4/ - Dual frequency
  V3/dataset5/ - Absorbent cotton
  V3/dataset6/ - WiFi mixed
  V3/dataset7/ - Environment

Requirements:
    pip install requests
    apt install aria2  (optional, for faster downloads)

Usage:
    python download_drffr2.py --email YOUR_EMAIL [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests


DATASET_ID = "b8a16448c1284fd1be1ded9ccc45be20"
API_URL = "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
DOWNLOAD_BASE = "https://china.scidb.cn/download"

SCENARIOS = [
    "/V3/dataset1",
    "/V3/dataset2",
    "/V3/dataset3",
    "/V3/dataset4",
    "/V3/dataset5",
    "/V3/dataset6",
    "/V3/dataset7",
]


def list_files_recursive(email, auth_token=None, path="/V3"):
    """Recursively list all files in the dataset."""
    headers = {
        "content-type": "application/json",
        "username": email,
        "traceid": email,
    }
    if auth_token:
        headers["authorization"] = auth_token

    all_files = []

    def _list(p):
        last_index = 0
        page_size = 200
        while True:
            body = {
                "dataSetId": DATASET_ID,
                "version": "V3",
                "path": p,
                "lastIndex": last_index,
                "pageSize": page_size,
            }
            try:
                resp = requests.post(API_URL, json=body, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json().get("data", [])
            except Exception as e:
                print(f"  API error for {p}: {e}")
                break

            if not data:
                break

            for item in data:
                if item.get("dir", False):
                    _list(item["path"])
                else:
                    all_files.append(item)

            last_index += len(data)
            if len(data) < page_size:
                break

    _list(path)
    return all_files


def download_file(file_id, email, output_path, max_retries=5):
    """Download a single file with retry logic."""
    url = f"{DOWNLOAD_BASE}?fileId={file_id}&username={email}&traceId={email}"

    for attempt in range(max_retries):
        try:
            # Try aria2c first
            result = subprocess.run(
                ["aria2c", "-x", "4", "-s", "4",
                 "--max-tries=3", "--retry-wait=10",
                 "--timeout=600", "--file-allocation=none",
                 "--continue=true",
                 "-d", os.path.dirname(output_path),
                 "-o", os.path.basename(output_path),
                 url],
                capture_output=True, text=True
            )
            if result.returncode == 0 and os.path.exists(output_path):
                return True
        except FileNotFoundError:
            pass

        # Fallback to requests
        try:
            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tmp_path = output_path + ".tmp"
            with open(tmp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
            os.rename(tmp_path, output_path)
            return True
        except Exception as e:
            print(f"    Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(10 * (attempt + 1))

    return False


def main():
    parser = argparse.ArgumentParser(description="Download DRFF-R2 dataset")
    parser.add_argument("--output-dir", default="data/drffr2",
                        help="Directory for downloaded files")
    parser.add_argument("--email", required=True,
                        help="Email for SciDB authentication")
    parser.add_argument("--token", default=None,
                        help="Optional JWT auth token")
    parser.add_argument("--file-list", default=None,
                        help="Path to pre-saved file list JSON (skip API listing)")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="Maximum concurrent downloads (SciDB rate limit)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get file list
    if args.file_list and os.path.exists(args.file_list):
        print(f"Loading file list from {args.file_list}")
        with open(args.file_list) as f:
            all_files = json.load(f)
    else:
        print("Listing all files via SciDB API (this may take a minute)...")
        all_files = list_files_recursive(args.email, args.token)

        # Save file list for resume
        file_list_path = os.path.join(args.output_dir, "drffr2_files.json")
        with open(file_list_path, 'w') as f:
            json.dump(all_files, f, indent=2)
        print(f"Saved file list ({len(all_files)} files) to {file_list_path}")

    mat_files = [f for f in all_files if f.get("fileName", "").endswith(".mat")]
    print(f"\nTotal .mat files to download: {len(mat_files)}")
    total_size_gb = sum(f.get("size", 0) for f in mat_files) / 1e9
    print(f"Total size: {total_size_gb:.1f} GB")

    # Download files
    downloaded = 0
    skipped = 0
    failed = 0

    for i, f in enumerate(mat_files, 1):
        fname = f["fileName"]
        fpath = f.get("path", "")

        # Reconstruct local path preserving directory structure
        rel_path = fpath.lstrip("/")
        local_path = os.path.join(args.output_dir, rel_path, fname)

        # Skip if already downloaded
        if os.path.exists(local_path):
            expected_size = f.get("size", 0)
            actual_size = os.path.getsize(local_path)
            if expected_size == 0 or abs(actual_size - expected_size) < 1024:
                skipped += 1
                continue

        size_mb = f.get("size", 0) / 1e6
        print(f"[{i}/{len(mat_files)}] {fname} ({size_mb:.0f} MB)...")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        success = download_file(f["id"], args.email, local_path)

        if success:
            downloaded += 1
        else:
            failed += 1
            print(f"  FAILED: {fname}")

    print(f"\nDownload summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {downloaded + skipped + failed}/{len(mat_files)}")


if __name__ == "__main__":
    main()
