#!/usr/bin/env bash
# Download DroneRFa dataset from SciDB China.
#
# DroneRFa: Dual-receiver companion to DroneRFb, 574 GB.
# Source: china.scidb.cn, fileId=c403fc76444e4b9989e4f3ff570f3b3d
#
# Requirements:
#   - aria2c (apt install aria2 / pacman -S aria2)
#   - unrar  (apt install unrar / pacman -S unrar)
#   - ~1.2 TB free disk space
#
# Usage:
#   bash download_droneRFa.sh [YOUR_EMAIL] [OUTPUT_DIR]

set -euo pipefail

EMAIL="${1:-your.email@example.com}"
OUTPUT_DIR="${2:-data/droneRFa}"
FILE_ID="c403fc76444e4b9989e4f3ff570f3b3d"

DOWNLOAD_URL="https://china.scidb.cn/download?fileId=${FILE_ID}&username=${EMAIL}&traceId=${EMAIL}"
RAR_FILE="${OUTPUT_DIR}/droneRFa.rar"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  DroneRFa Dataset Download"
echo "  Size: ~574 GB"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

if [ "${EMAIL}" = "your.email@example.com" ]; then
    echo ""
    echo "ERROR: Please provide your SciDB registration email."
    echo "Usage: bash download_droneRFa.sh your.email@example.com [output_dir]"
    exit 1
fi

# Check dependencies
if ! command -v aria2c &> /dev/null; then
    echo "ERROR: aria2c not found. Install with: apt install aria2"
    exit 1
fi

# Download
echo ""
echo "Downloading DroneRFa (~574 GB)..."
echo "This will take several hours depending on your connection."
echo ""

aria2c \
    -x 4 \
    -s 4 \
    --max-tries=0 \
    --retry-wait=30 \
    --timeout=600 \
    --file-allocation=none \
    --continue=true \
    -d "${OUTPUT_DIR}" \
    -o "droneRFa.rar" \
    "${DOWNLOAD_URL}"

echo ""
echo "Download complete: ${RAR_FILE}"
echo "File size: $(du -sh "${RAR_FILE}" | cut -f1)"

# Extract
echo ""
echo "Extracting (this will take a while)..."
cd "${OUTPUT_DIR}"
unrar x -o- droneRFa.rar .

echo ""
echo "Extraction complete."
echo "Contents:"
ls -la "${OUTPUT_DIR}/"
