# GPU Request Scheduling Simulator

A comprehensive simulation framework for evaluating GPU request scheduling strategies in large language model (LLM) serving systems. This project provides a flexible environment to compare different scheduling policies, analyze performance metrics, and conduct parameter sweeps.

## Features

- **Multiple Scheduling Strategies**: Implementations of FCFS, JSQ, BF-IO, Round Robin, LPT, and more
- **Real Dataset Support**: Load requests from ShareGPT, ArXiv, BurstGPT, or custom CSV/JSONL files
- **Comprehensive Metrics**: Throughput, latency, GPU imbalance, utilization, power consumption, and energy
- **Batch Experiments**: Automated scripts for parameter sweeps (batch size, GPU count, horizon values)
- **Visualization**: GPU load trajectory plots and strategy comparison charts
- **Hyperparameter Optimization**: Optuna-based search for optimal strategy parameters

## Project Structure

```
.
├── simlib/                          # Core simulation library
│   ├── __init__.py
│   ├── scheduler.py                # SchedulerSim - main simulation engine
│   ├── request_generator.py         # Request generation and dataset loaders
│   ├── efficiency.py                # Efficiency function registry
│   └── strategies/                   # Scheduling strategy implementations
│       ├── strategy_FCFS.py
│       ├── strategy_join_shortest_queue.py
│       ├── strategy_balance_future.py      # BF-IO strategy
│       ├── strategy_balance_future_h0.py  # BF-IO with H=0
│       ├── strategy_now.py
│       ├── strategy_round_robin.py
│       ├── strategy_LPT_heap.py
│       └── strategy_align_current_max.py
├── scripts/                          # Executable scripts
│   ├── main.py                      # Main simulation script with visualization
│   ├── optimize.py                   # Optuna hyperparameter search
│   ├── lp_balance_solver.py         # MILP solver example
│   ├── sweep_balance_future_delta.py # Delta parameter sweep
│   ├── test_balance_future_horizon.py
│   └── bash_*.sh                    # Batch experiment scripts
│       ├── bash_multiB.sh            # Batch size sweep
│       ├── bash_multiG.sh             # GPU count sweep
│       └── bash_multiH.sh             # Horizon value sweep
├── arxiv_data_8000.json              # Sample ArXiv dataset
├── burst_data_8000.json              # Sample BurstGPT dataset
├── longbench_data_8000.json           # Sample LongBench dataset
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
cd Load

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.22
pandas>=1.5
matplotlib>=3.6
joblib>=1.2
tqdm>=4.64
optuna>=3.0
```

## Quick Start

### Basic Simulation

Run a simulation with default parameters:

```bash
python scripts/main.py
```

This will:
- Generate random requests
- Run multiple strategies (FCFS, JSQ, BF-IO variants)
- Generate comparison plots
- Output metrics to console and CSV

### Run Specific Strategy

```bash
python scripts/main.py --strategy BF-IO --N 50000 --G 32 --B 72
```

### Run with Real Dataset

```bash
python scripts/main.py \
    --load_from_real_dataset \
    --dataset_path burst_data_8000.json \
    --strategy BF-IO \
    --H 80
```

### Batch Experiments

Run parameter sweeps using bash scripts:

```bash
# Sweep batch sizes
bash scripts/bash_multiB.sh --strategy BF-IO --G 256 --H 140

# Sweep GPU counts
bash scripts/bash_multiG.sh --strategy BF-IO --B 72 --H 60

# Sweep horizon values
bash scripts/bash_multiH.sh
```

## Command Line Arguments

### Main Script (`scripts/main.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--N` | int | 100000 | Number of requests |
| `--n_repeat` | int | 100 | Number of experiment repetitions |
| `--seed` | int | 42 | Random seed |
| `--n_jobs` | int | -1 | Parallel jobs (-1 = all cores) |
| `--delta_lower` | float | 0.0 | Lower bound for request generation |
| `--delta_upper` | float | 0.0 | Upper bound for request generation |
| `--load_from_real_dataset` | flag | False | Load from real dataset file |
| `--dataset_path` | str | None | Path to dataset file |
| `--strategy` | str | None | Strategy to run: FCFS, JSQ, BF-IO, BF-IO-H0 |
| `--H` | int | None | Horizon value for BF-IO strategy |
| `--B` | int | 72 | Batch size (max concurrent requests per GPU) |
| `--G` | int | 16 | Number of GPUs (workers) |
| `--skip_plot` | flag | False | Skip plotting, only run statistics |

### Example Commands

```bash
# Compare all strategies with 50K requests
python scripts/main.py --N 50000 --n_repeat 10

# Run BF-IO with specific horizon
python scripts/main.py --strategy BF-IO --H 80 --N 80000

# Run FCFS with custom GPU and batch size
python scripts/main.py --strategy FCFS --G 64 --B 96

# Use real dataset with BF-IO
python scripts/main.py \
    --load_from_real_dataset \
    --dataset_path longbench_data_8000.json \
    --strategy BF-IO \
    --H 140 \
    --G 256 \
    --B 72
```

## Scheduling Strategies

### Available Strategies

1. **FCFS (First-Come-First-Served)**
   - Simple queue-based scheduling
   - Baseline strategy

2. **JSQ (Join Shortest Queue)**
   - Assigns requests to GPU with minimum current load
   - Load balancing heuristic

3. **BF-IO (Balance Future - Input/Output)**
   - Two-phase strategy with future load prediction
   - Greedy phase + fine-grained subset selection
   - Configurable horizon (H) for prediction window

4. **BF-IO-H0**
   - BF-IO variant with H=0 (no future prediction)
   - Only considers current load

5. **Round Robin**
   - Cyclic assignment to GPUs

6. **LPT (Longest Processing Time)**
   - Heap-based longest-first scheduling

7. **Balance Now**
   - Current load balancing without future prediction

8. **Align Current Max**
   - Aligns to current maximum load

### Strategy Selection

```python
# In code
from simlib.strategies import (
    policy_FCFS,
    policy_join_shortest_queue,
    policy_balance_future,
    policy_balance_future_h0,
    policy_round_robin,
    policy_lpt_heap,
    policy_balance_now,
    policy_align_current_max,
)
```

## Metrics

The simulator computes the following performance metrics:

- **Makespan**: Total time for all requests to complete
- **Average Latency**: Mean completion time per request
- **Throughput**: Tokens processed per second
- **QPM (Queries Per Minute)**: Completed requests per minute
- **Average GPU Imbalance**: Load imbalance across GPUs
  - Formula: `avg_imbalance = mean(n_worker * max_load - sum_load)`
- **Average GPU Utilization**: Mean GPU utilization rate
- **GPU Idle Rate**: Fraction of time GPUs are idle
- **Average Power**: Mean power consumption (Watts)
- **Total Energy**: Total energy consumption (Joules)

### Power Model

The simulator includes a power consumption model based on GPU utilization:
- Idle power: 100W
- Peak power: 400W
- Power scales sublinearly with utilization (γ = 0.7)

## Dataset Formats

### Supported Formats

1. **Random Generation** (default)
   - Synthetic requests with configurable parameters

2. **ShareGPT Format**
   - JSON files with conversation data
   - Extracts input/output token counts

3. **ArXiv Format**
   - JSON files with paper metadata
   - `data` field containing request information

4. **BurstGPT Format**
   - JSON files with burst request patterns

5. **CSV Format**
   - Columns: `input_tokens`, `output_tokens`
   - Used by `main_csv.py`

6. **JSONL Format**
   - One JSON object per line
   - Used by `main_jsonl.py`

### Loading Datasets

```python
from simlib.request_generator import (
    load_sharegpt_dataset,
    load_arxiv_dataset,
    load_burstgpt_dataset,
)

# Load dataset
s_arr, o_arr, o_est_arr = load_arxiv_dataset(
    "arxiv_data_8000.json",
    N=50000,
    rng=np.random.default_rng(42),
    delta_lower=0.0,
    delta_upper=0.0,
)
```

## Batch Experiment Scripts

The repository includes bash scripts for automated parameter sweeps:

### `bash_multiB.sh`
Sweeps batch size (B) values:
```bash
bash scripts/bash_multiB.sh --strategy BF-IO --G 256 --H 140
```

### `bash_multiG.sh`
Sweeps GPU count (G) values:
```bash
bash scripts/bash_multiG.sh --strategy BF-IO --B 72 --H 60
```

### `bash_multiH.sh`
Sweeps horizon (H) values:
```bash
bash scripts/bash_multiH.sh
```

### Output

Results are saved to `results/` directory:
- `multiB_results.csv` - Batch size sweep results
- `multiG_results.csv` - GPU count sweep results
- `multiH_results.csv` - Horizon sweep results
- `power_*.csv` - Power consumption data
- `power_timeseries_*.csv` - Time-series power data
- `*_output.log` - Experiment logs


## Hyperparameter Optimization

Use Optuna for hyperparameter search:

```bash
python scripts/optimize.py
```

This searches for optimal parameters in `strategy_balance_future.py` to minimize GPU imbalance.

### Adding New Strategies

1. Create `simlib/strategies/strategy_xxx.py`
2. Implement `policy_xxx(sim)` function
3. Export in `simlib/strategies/__init__.py`
4. Add to strategy map in `scripts/main.py`

### Code Structure

- `simlib/scheduler.py`: Core simulation engine
- `simlib/request_generator.py`: Request generation and dataset loaders
- `simlib/strategies/`: Strategy implementations
- `scripts/main.py`: Main entry point with CLI
- `scripts/bash_*.sh`: Batch experiment automation

