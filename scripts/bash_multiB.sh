#!/bin/bash

# bash_multiB.sh - Test the impact of different batch_size (B) values on BF-IO strategy
# Observe changes in Imbalance, Throughput, and TPOT

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# Default parameters
N=80000
N_REPEAT=1  # Run experiment once for each B value
SEED=42
N_JOBS=-1
DELTA_LOWER=0.0
DELTA_UPPER=0.0
LOAD_FROM_REAL_DATASET=true
DATASET_PATH="longbench_data_8000.json"
SKIP_PLOT=true  # Skip plotting, only run statistical experiments
G=256  # Fixed GPU count
H=140  # Fixed H value
STRATEGY="BF-IO"  # Default strategy: BF-IO, FCFS, JSQ, BF-IO-H0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --N)
            N="$2"
            shift 2
            ;;
        --n_repeat)
            N_REPEAT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --G)
            G="$2"
            shift 2
            ;;
        --H)
            H="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --strategy STRATEGY    Routing strategy (BF-IO, FCFS, JSQ, BF-IO-H0) [default: BF-IO]"
            echo "  --N N                  Number of requests [default: 100000]"
            echo "  --n_repeat N           Number of experiment repetitions [default: 1]"
            echo "  --seed SEED            Random seed [default: 42]"
            echo "  --G G                  Number of GPUs [default: 16]"
            echo "  --H H                  HORIZON_MAX value [default: 80]"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --strategy FCFS"
            echo "  $0 --strategy BF-IO --G 32"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help to see help information"
            exit 1
            ;;
    esac
done

# B value list (can be manually modified)
B_VALUES=(72)

# Output files
OUTPUT_DIR="results"
# Select output filename based on strategy
if [ "$STRATEGY" = "FCFS" ]; then
    OUTPUT_FILE="$OUTPUT_DIR/multiB_FCFS_results.csv"
elif [ "$STRATEGY" != "BF-IO" ]; then
    OUTPUT_FILE="$OUTPUT_DIR/multiB_${STRATEGY}_results.csv"
else
    OUTPUT_FILE="$OUTPUT_DIR/multiB_results.csv"
fi
mkdir -p "$OUTPUT_DIR"

# Delete old CSV file if it exists, main.py will create a new one
rm -f "$OUTPUT_FILE"

echo "=========================================="
echo "$STRATEGY Multi-B (batch_size) Experiment"
echo "=========================================="
echo "Parameter settings:"
echo "  Strategy: $STRATEGY"
echo "  N: $N"
echo "  n_repeat: $N_REPEAT"
echo "  seed: $SEED"
echo "  G (n_worker): $G"
echo "  H: $H"
echo "  B value list: ${B_VALUES[*]}"
echo "=========================================="
echo ""

# Loop through different B values
for B in "${B_VALUES[@]}"; do
    echo "----------------------------------------"
    echo "Running experiment for B=$B..."
    echo "----------------------------------------"
    
    # Run experiment
    python3 scripts/main.py \
        --N "$N" \
        --n_repeat "$N_REPEAT" \
        --seed "$SEED" \
        --n_jobs "$N_JOBS" \
        --delta_lower "$DELTA_LOWER" \
        --delta_upper "$DELTA_UPPER" \
        --strategy "$STRATEGY" \
        --H "$H" \
        --B "$B" \
        --G "$G" \
        --skip_plot \
        $([ "$LOAD_FROM_REAL_DATASET" = true ] && echo "--load_from_real_dataset") \
        --dataset_path "$DATASET_PATH" 2>&1 | tee "$OUTPUT_DIR/B${B}_${STRATEGY}_output.log"
    
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

