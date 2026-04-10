#!/bin/bash
# =============================================================================
# RFML-MoE: Full Pipeline Runner
# Drone RF Signal Detection with Multi-modal Mixture of Experts
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-configs/default.yaml}"
LOG_LEVEL="${2:-INFO}"

echo "============================================="
echo "  RFML-MoE Pipeline"
echo "  Config: $CONFIG"
echo "============================================="

# ROCm environment setup (MI300X optimization)
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=2
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3

# Step 1: Show system info
echo ""
echo "[Step 1/6] System Information"
python main.py --config "$CONFIG" info

# Step 2: Download datasets
echo ""
echo "[Step 2/6] Downloading Datasets"
python main.py --config "$CONFIG" --log-level "$LOG_LEVEL" download

# Step 3: Preprocess raw data
echo ""
echo "[Step 3/6] Preprocessing Raw Data"
python scripts/preprocess.py

# Step 4: Generate synthetic data
echo ""
echo "[Step 4/7] Generating Synthetic Training Data"
python main.py --config "$CONFIG" --log-level "$LOG_LEVEL" generate-synthetic \
    --num-samples 100000 --difficulty mixed

# Step 5: Extract features and create shards
echo ""
echo "[Step 5/7] Feature Extraction + Shard Creation"
python main.py --config "$CONFIG" --log-level "$LOG_LEVEL" extract-features
python scripts/create_shards.py

# Step 6: Train model (4-phase progressive training)
echo ""
echo "[Step 6/7] Training MoE Model (4-phase progressive)"
python main.py --config "$CONFIG" --log-level "$LOG_LEVEL" train --phase all

# Step 7: Evaluate
echo ""
echo "[Step 7/7] Evaluation"
BEST_CKPT=$(ls -t checkpoints/*.pt 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    python main.py --config "$CONFIG" --log-level "$LOG_LEVEL" evaluate \
        --checkpoint "$BEST_CKPT" \
        --output results \
        --open-set
else
    echo "WARNING: No checkpoint found. Skipping evaluation."
fi

echo ""
echo "============================================="
echo "  Pipeline Complete!"
echo "  Results: results/"
echo "============================================="
