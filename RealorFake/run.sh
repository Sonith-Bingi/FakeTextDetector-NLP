#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- RealOrFake Pipeline Started ---"

# Step 1: Install dependencies
echo "[Step 1/3] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 2: Run the main training and prediction pipeline
# This script will handle unzipping data, feature engineering,
# model training, and submission file generation.
echo "[Step 2/3] Running main pipeline (src/main.py)..."
python3 -m src.main

# Step 3: Done
echo "[Step 3/3] Pipeline complete."
echo "Submission file created at: ./submission.csv"
echo "--- All steps finished successfully! ---"