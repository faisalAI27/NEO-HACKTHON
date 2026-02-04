#!/bin/bash
set -e

# Configuration
MAX_EPOCHS=30
BATCH_SIZE=8

echo "=================================================================="
echo "STARTING MOSAIC EXPERIMENTS"
echo "=================================================================="
echo "Max Epochs: $MAX_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=================================================================="

# Create log dir
mkdir -p logs

# 1. Unimodal Baselines
echo "[1/4] Running Clinical Baseline..."
for fold in {0..4}; do
    echo "  > Fold $fold"
    python3 scripts/04_train_unimodal.py --modality clinical --fold $fold --max_epochs $MAX_EPOCHS --batch_size $BATCH_SIZE > logs/clinical_fold${fold}.log 2>&1
done

echo "[2/4] Running RNA Baseline..."
for fold in {0..4}; do
    echo "  > Fold $fold"
    python3 scripts/04_train_unimodal.py --modality rna --fold $fold --max_epochs $MAX_EPOCHS --batch_size $BATCH_SIZE > logs/rna_fold${fold}.log 2>&1
done

echo "[3/4] Running WSI Baseline..."
for fold in {0..4}; do
    echo "  > Fold $fold"
    python3 scripts/04_train_unimodal.py --modality wsi --fold $fold --max_epochs $MAX_EPOCHS --batch_size $BATCH_SIZE > logs/wsi_fold${fold}.log 2>&1
done

# 2. Multimodal Model
echo "[4/4] Running Multimodal MOSAIC..."
for fold in {0..4}; do
    echo "  > Fold $fold"
    python3 scripts/05_train_multimodal.py --fold $fold --max_epochs $MAX_EPOCHS --batch_size $BATCH_SIZE > logs/mosaic_fold${fold}.log 2>&1
done

echo "=================================================================="
echo "EXPERIMENTS COMPLETED"
echo "Check logs/ directory for details."
echo "=================================================================="
