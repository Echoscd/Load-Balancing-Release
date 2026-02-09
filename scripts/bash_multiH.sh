#!/bin/bash

# bash_multiH.sh - Test the impact of different H values on BF-IO strategy
# Observe changes in Imbalance, Throughput, and TPOT

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# Default parameters
N=100000
N_REPEAT=1  # Run experiment once for each H value
SEED=42
N_JOBS=-1
DELTA_LOWER=0.0
DELTA_UPPER=0.0
LOAD_FROM_REAL_DATASET=true
DATASET_PATH="burst_data_8000.json"
SKIP_PLOT=true  # Skip plotting, only run statistical experiments

# H value list (can be manually modified)
H_VALUES=(0 10 20 40 60 80 120 200)

# Output files
OUTPUT_DIR="results"
OUTPUT_FILE="$OUTPUT_DIR/multiH_results.csv"
mkdir -p "$OUTPUT_DIR"

# Delete old CSV file if it exists, main.py will create a new one
rm -f "$OUTPUT_FILE"

echo "=========================================="
echo "BF-IO Multi-H Experiment"
echo "=========================================="
echo "Parameter settings:"
echo "  N: $N"
echo "  n_repeat: $N_REPEAT"
echo "  seed: $SEED"
echo "  H value list: ${H_VALUES[*]}"
echo "=========================================="
echo ""

# Loop through different H values
for H in "${H_VALUES[@]}"; do
    echo "----------------------------------------"
    echo "Running experiment for H=$H..."
    echo "----------------------------------------"
    
    # Run experiment
    python3 scripts/main.py \
        --N "$N" \
        --n_repeat "$N_REPEAT" \
        --seed "$SEED" \
        --n_jobs "$N_JOBS" \
        --delta_lower "$DELTA_LOWER" \
        --delta_upper "$DELTA_UPPER" \
        --H "$H" \
        --skip_plot \
        $([ "$LOAD_FROM_REAL_DATASET" = true ] && echo "--load_from_real_dataset") \
        --dataset_path "$DATASET_PATH" 2>&1 | tee "$OUTPUT_DIR/H${H}_output.log"
    
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

