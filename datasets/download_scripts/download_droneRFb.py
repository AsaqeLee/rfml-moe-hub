#!/usr/bin/env python3
"""
Download DroneRFb-DIR dataset from SciDB China.

DroneRFb-DIR: 13 classes (6 drone types x 2 individuals + background), 65 GB.
Split-zip format: 32 parts that must be combined before extraction.
Source: china.scidb.cn, dataSetId=84cf9101e739402784b1396783881202

Requirements:
    pip install requests
    apt install unzip

Usage:
    python download_droneRFb.py [--output-dir OUTPUT_DIR] --email YOUR_EMAIL
"""

import argparse
import json
import os
import subprocess
import sys

import requests


DATASET_ID = "84cf9101e739402784b1396783881202"
API_URL = "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
DOWNLOAD_BASE = "https://china.scidb.cn/download"


def list_files(email, auth_token=None, path="/"):
    """List files in the dataset via SciDB API."""
    headers = {
        "content-type": "application/json",
        "username": email,
        "traceid": email,
    }
    if auth_token:
        headers["authorization"] = auth_token

    all_files = []
    last_index = 0
    page_size = 200

    while True:
        body = {
            "dataSetId": DATASET_ID,
            "version": "V3",
            "path": path,
            "lastIndex": last_index,
            "pageSize": page_size,
        }
        resp = requests.post(API_URL, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            break

        all_files.extend(data)
        last_index += len(data)

        if len(data) < page_size:
            break

    return all_files


def download_file(file_id, email, output_path, max_retries=3):
    """Download a single file from SciDB."""
    url = f"{DOWNLOAD_BASE}?fileId={file_id}&username={email}&traceId={email}"

    for attempt in range(max_retries):
        try:
            # Use aria2c if available for better performance
            result = subprocess.run(
                ["aria2c", "-x", "4", "-s", "4", "--max-tries=3",
                 "--retry-wait=10", "--timeout=600",
                 "-o", output_path, url],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            # aria2c not available, use requests
            pass

        try:
            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")

    return False


def main():
    parser = argparse.ArgumentParser(description="Download DroneRFb-DIR dataset")
    parser.add_argument("--output-dir", default="data/droneRFb",
                        help="Directory for downloaded files")
    parser.add_argument("--email", required=True,
                        help="Email for SciDB authentication")
    parser.add_argument("--token", default=None,
                        help="Optional JWT auth token")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Listing dataset files...")
    files = list_files(args.email, args.token)
    zip_files = [f for f in files if f.get("fileName", "").endswith(".zip")]
    print(f"Found {len(zip_files)} zip parts")

    # Download all zip parts
    for i, f in enumerate(sorted(zip_files, key=lambda x: x["fileName"]), 1):
        fname = f["fileName"]
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) == f.get("size", 0):
            print(f"[{i}/{len(zip_files)}] {fname} -- already downloaded, skipping")
            continue
        print(f"[{i}/{len(zip_files)}] Downloading {fname} ({f.get('size', 0) / 1e9:.1f} GB)...")
        success = download_file(f["id"], args.email, fpath)
        if not success:
            print(f"  FAILED to download {fname}")

    # Combine and extract
    print("\nCombining split zip files...")
    combined = os.path.join(args.output_dir, "twin_droneRF_combined.zip")
    parts = sorted(
        os.path.join(args.output_dir, f)
        for f in os.listdir(args.output_dir)
        if f.startswith("twin_droneRF.zip")
    )
    if parts and not os.path.exists(combined):
        cmd = f"cat {' '.join(parts)} > {combined}"
        os.system(cmd)
        print(f"Combined {len(parts)} parts into {combined}")

    print("\nExtracting...")
    subprocess.run(["unzip", "-o", combined, "-d", args.output_dir])

    # Verify
    train_dir = os.path.join(args.output_dir, "twin_droneRF", "train")
    test_dir = os.path.join(args.output_dir, "twin_droneRF", "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        n_train = len([f for f in os.listdir(train_dir) if f.endswith(".mat")])
        n_test = len([f for f in os.listdir(test_dir) if f.endswith(".mat")])
        print(f"\nVerification: {n_train} train files, {n_test} test files")
        print(f"Expected: 2177 train, 2513 test")


if __name__ == "__main__":
    main()
