from __future__ import annotations

import sys
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simlib import SchedulerSim, generate_requests
from simlib.request_generator import load_sharegpt_dataset, load_arxiv_dataset, load_burstgpt_dataset
from simlib.strategies import (
    policy_balance_future,
    policy_balance_now,
    policy_FCFS,
    policy_join_shortest_queue,
    policy_lpt_heap,
    policy_align_current_max,
    policy_round_robin,
    policy_balance_future_h0,
)
from simlib.strategies import strategy_balance_future as balance_future_mod


def normalize_strategy_name(strategy_name: str) -> str:
    """
    Normalize strategy name for use in filenames.
    Examples: "BF-IO(H=80)" -> "BF_IO_H_80"
              "FCFS" -> "FCFS"
              "Join Shortest Queue" -> "JSQ"
    """
    # Replace spaces and special characters
    name = strategy_name.replace(" ", "_")
    name = name.replace("-", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace("=", "_")
    # Simplify common strategy names
    if "Join_Shortest_Queue" in name:
        name = name.replace("Join_Shortest_Queue", "JSQ")
    return name

@contextmanager
def override_horizon(h: int):
    """Temporarily modify the HORIZON_MAX constant in the strategy."""
    original = balance_future_mod.HORIZON_MAX
    balance_future_mod.HORIZON_MAX = int(h)
    try:
        yield
    finally:
        balance_future_mod.HORIZON_MAX = original

# ------------------- Combined function: returns sim and metrics -------------------
def run_sim_and_metrics(
    policy_fn: Callable[[SchedulerSim], None],
    eff_name: str,
    N: int,
    seed: int,
    delta_lower: float,
    delta_upper: float,
    load_from_real_dataset: bool = False,
    dataset_path: Optional[str] = None,
    n_worker: int = 16,
    batch_size: int = 72,
) -> Tuple[SchedulerSim, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    
    if load_from_real_dataset:
        if dataset_path is None:
            # Default to local burst data file
            dataset_path = str(ROOT / "burst_data_8000.json")
        
        # Check file path, use load_arxiv_dataset if ArXiv format
        dataset_path_obj = Path(dataset_path)
        if dataset_path_obj.suffix == '.json' and dataset_path_obj.exists():
            # Try to detect if it's ArXiv format
            try:
                import json
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    sample = json.load(f)
                    if isinstance(sample, dict) and 'data' in sample:
                        # ArXiv format
                        s_arr, o_arr, o_est_arr = load_arxiv_dataset(
                            dataset_path,
                            N,
                            rng=rng,
                            delta_lower=delta_lower,
                            delta_upper=delta_upper,
                        )
                    else:
                        # ShareGPT format
                        s_arr, o_arr, o_est_arr = load_sharegpt_dataset(
                            dataset_path,
                            N,
                            rng=rng,
                            delta_lower=delta_lower,
                            delta_upper=delta_upper,
                        )
            except Exception:
                # If detection fails, try as ShareGPT format
                s_arr, o_arr, o_est_arr = load_sharegpt_dataset(
                    dataset_path,
                    N,
                    rng=rng,
                    delta_lower=delta_lower,
                    delta_upper=delta_upper,
                )
        elif dataset_path_obj.is_dir():
            # Directory, use ShareGPT loader
            s_arr, o_arr, o_est_arr = load_sharegpt_dataset(
                dataset_path,
                N,
                rng=rng,
                delta_lower=delta_lower,
                delta_upper=delta_upper,
            )
        else:
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    else:
        s_arr, o_arr, o_est_arr = generate_requests(
            N,
            rng=rng,
            delta_lower=delta_lower,
            delta_upper=delta_upper,
        )
    sim = SchedulerSim(
        n_worker=n_worker,
        batch_size=batch_size,
        C=(batch_size * 0.6e-3 + 35e-3) / 16,
        t_length=1.5 * 2 * 576 * 61e-6 / 2**20,
        s_list=s_arr,
        o_list=o_arr,
        o_est_list=o_est_arr,
        policy_fn=policy_fn,
        eff_name=eff_name,
        record_history=True,
    )
    metrics = sim.run(verbose=False)
    return sim, metrics

# ------------------- Plotting functions -------------------
def plot_gpu_tokens(sim: SchedulerSim, title: str = ""):
    hist = np.vstack(sim.hist_sum_lengths[:4000])   # (num_steps, n_worker)
    steps = np.array(sim.hist_steps[:4000])         # (num_steps,)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, sim.n_worker))
    for gid in range(sim.n_worker):
        ax.plot(
            steps,
            hist[:, gid],
            label=f"Worker {gid+1}",
            color=colors[gid],
            linewidth=2.4,
            alpha=0.9,
        )
    ax.set_xlabel("Decode step", fontsize=21, fontweight="semibold", color="#1f1f2e")
    ax.set_ylabel("Total token load per worker", fontsize=21, fontweight="semibold", color="#1f1f2e")
    ax.tick_params(axis="both", labelsize=18, colors="#2b2b3c", width=2.0)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, _: f"{val / 1000:.0f}k")
    )
    ax.legend(
        fontsize=16,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.3,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderpad=0.8,
    )
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35, color="#7c8aa6")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2.2)
        ax.spines[spine].set_color("#1f1f2e")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("#f4f6fb")
    fig.patch.set_facecolor("#eef1f7")
    fig.tight_layout(rect=[0, 0, 1, 1])
    algo_name = (title.split(" (")[0] if title else "worker_plot").replace(" ", "_")
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{algo_name}.jpg", dpi=600, format='jpg')
    plt.show()


def compute_metrics_from_sim(sim: SchedulerSim) -> Dict[str, float]:
    """Compute metrics from sim object (does not modify sim state)"""
    # Note: sim.run() has already been executed, cannot call it again here
    # So we need to recompute from historical data or use already computed metrics
    
    # Calculate Average Imbalance
    # Formula: AvgImbalance = (1/K) * Σ_{k=1}^{K} Imbalance(k)
    # This averages over steps K, not over time
    if len(sim.hist_sum_lengths) > 0:
        hist = np.vstack(sim.hist_sum_lengths)
        max_loads = hist.max(axis=1)
        sum_loads = hist.sum(axis=1)
        imbalances = sim.n_worker * max_loads - sum_loads
        # Average over steps (not over time)
        avg_imbalance = float(imbalances.mean())
    else:
        avg_imbalance = 0.0
    
    # Calculate Throughput: tokens / time
    # Computed from sim's finish_time and start_time
    valid = sim.finish_time > 0
    if valid.any():
        total_tokens = int(sim.o_arr[valid].sum())
        total_time = float(sim.finish_time[valid].max() - sim.start_time[valid].min())
        throughput = total_tokens / total_time if total_time > 0 else 0.0
    else:
        throughput = 0.0
    
    # Calculate TPOT: average of (finish_time - start_time) / o for each request
    valid = (sim.finish_time > 0) & (sim.o_arr > 0)
    if valid.any():
        latencies = (sim.finish_time[valid] - sim.start_time[valid]) / sim.o_arr[valid]
        tpot = float(latencies.mean())
    else:
        tpot = 0.0
    
    return {
        "avg_imbalance": avg_imbalance,
        "throughput": throughput,
        "tpot": tpot,
        "idle_rate": 0.0,  # Cannot recompute from historical data, use default
        "energy": 0.0,  # Cannot recompute from historical data, use default
        "power": 100.0,  # Default to P_idle
    }


def plot_gpu_tokens_subplot(ax, sim: SchedulerSim, title: str = "", metrics: Optional[Dict[str, float]] = None):
    """Plot GPU tokens on given subplot and display metrics"""
    hist = np.vstack(sim.hist_sum_lengths[:4000])   # (num_steps, n_worker)
    steps = np.array(sim.hist_steps[:4000])         # (num_steps,)
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, sim.n_worker))
    for gid in range(sim.n_worker):
        ax.plot(
            steps,
            hist[:, gid],
            label=f"Worker {gid+1}",
            color=colors[gid],
            linewidth=1.8,
            alpha=0.85,
        )
    ax.set_xlabel("Decode step", fontsize=16, fontweight="semibold", color="#1f1f2e")
    ax.set_ylabel("Total token load per worker", fontsize=16, fontweight="semibold", color="#1f1f2e")
    ax.set_title(title, fontsize=18, fontweight="bold", color="#1f1f2e", pad=10)
    ax.tick_params(axis="both", labelsize=14, colors="#2b2b3c", width=1.5)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, _: f"{val / 1000:.0f}k")
    )
    ax.legend(
        fontsize=9,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderpad=0.4,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.5,
    )
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.3, color="#7c8aa6")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_color("#1f1f2e")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("#f4f6fb")
    
    # Display metrics
    if metrics is None:
        computed_metrics = compute_metrics_from_sim(sim)
    else:
        # Extract from passed metrics (metrics come from sim.run() return value)
        computed_metrics = {
            "avg_imbalance": metrics.get('avg_gpu_imbalance', 0.0),
            "throughput": metrics.get('throughput_tokens_per_sec', 0.0),
            "tpot": metrics.get('avg_latency_per_token', 0.0),
            "idle_rate": metrics.get('avg_gpu_idle_rate', 0.0),
            "energy": metrics.get('total_energy', 0.0),
            "power": metrics.get('avg_power', 0.0),
        }
    
    # Display metrics in top-left corner of plot
    metrics_text = (
        f"Avg Imbalance: {computed_metrics['avg_imbalance']:.1f}\n"
        f"Throughput: {computed_metrics['throughput']:.2f} tok/s\n"
        f"TPOT: {computed_metrics['tpot']:.4f} s/tok\n"
        f"GPU Idle Rate: {computed_metrics['idle_rate']:.4f}\n"
        f"Avg Power: {computed_metrics['power']:.1f} W\n"
        f"Total Energy: {computed_metrics['energy']:.2f} J"
    )
    ax.text(
        0.02, 0.98, metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
        family='monospace'
    )


def plot_four_strategies_comparison(strategies_dict: Dict[str, SchedulerSim], eff_name: str = ""):
    """Plot subplots of four strategies in one figure"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    
    strategy_names = list(strategies_dict.keys())
    for idx, (strat_name, sim_data) in enumerate(strategies_dict.items()):
        if idx >= 4:
            break
        if isinstance(sim_data, tuple):
            sim, metrics = sim_data
        else:
            sim = sim_data
            metrics = None
        plot_gpu_tokens_subplot(axes[idx], sim, title=strat_name, metrics=metrics)
    
    fig.patch.set_facecolor("#eef1f7")
    fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=3, w_pad=3)
    
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"four_strategies_comparison.jpg", dpi=600, format='jpg')
    plt.show()
    plt.close(fig)

# ------------------- Main program -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scheduling strategy simulation experiments")
    parser.add_argument("--N", type=int, default=100000, help="Number of requests (default: 100000)")
    parser.add_argument("--n_repeat", type=int, default=100, help="Number of experiment repetitions (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs, -1 means use all cores (default: -1)")
    parser.add_argument("--delta_lower", type=float, default=0.0, help="delta_lower parameter (default: 0.0)")
    parser.add_argument("--delta_upper", type=float, default=0.0, help="delta_upper parameter (default: 0.0)")
    parser.add_argument("--load_from_real_dataset", action="store_true", help="Load from real dataset")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--H", type=int, default=None, help="HORIZON_MAX value for BF-IO strategy (H parameter), if specified only runs BF-IO strategy")
    parser.add_argument("--B", type=int, default=72, help="batch_size parameter (B parameter), max concurrent requests per GPU (default: 72)")
    parser.add_argument("--G", type=int, default=16, help="n_worker parameter (G parameter), number of GPUs (default: 16)")
    parser.add_argument("--strategy", type=str, default=None, choices=["FCFS", "JSQ", "BF-IO", "BF-IO-H0"], 
                       help="Specify strategy to run (FCFS, JSQ, BF-IO, BF-IO-H0)")
    parser.add_argument("--skip_plot", action="store_true", help="Skip plotting, only run statistical experiments")
    
    args = parser.parse_args()
    
    N = args.N
    n_repeat = args.n_repeat
    global_seed = args.seed
    n_jobs = args.n_jobs
    delta_lower = args.delta_lower
    delta_upper = args.delta_upper
    load_from_real_dataset = args.load_from_real_dataset
    dataset_path = args.dataset_path if args.dataset_path else str(ROOT / "burst_data_8000.json")
    H_value = args.H
    batch_size = args.B
    n_worker = args.G
    skip_plot = args.skip_plot

    # Strategy mapping
    strategy_map = {
        "FCFS": ("FCFS", policy_FCFS),
        "JSQ": ("Join Shortest Queue", policy_join_shortest_queue),
        "BF-IO": ("BF-IO", policy_balance_future),
        "BF-IO-H0": ("BF-IO(H=0)", policy_balance_future_h0),
    }
    
    # Strategy selection logic: --strategy has higher priority than --H
    if args.strategy is not None:
        # If strategy is specified, only run that strategy
        n_repeat = 1  # Run experiment once for each parameter value
        strat_name, strat_fn = strategy_map[args.strategy]
        # If H value is specified and strategy is BF-IO related, use H value
        if H_value is not None and args.strategy in ["BF-IO", "BF-IO-H0"]:
            comparison_strategies = {
                f"{strat_name}(H={H_value})": strat_fn,
            }
        else:
            comparison_strategies = {
                strat_name: strat_fn,
            }
    elif H_value is not None:
        # If only H value is specified (no strategy specified), default to BF-IO strategy
        n_repeat = 1  # Run experiment once for each H value
        # Use override_horizon to set H value
        with override_horizon(H_value):
            comparison_strategies = {
                f"BF-IO(H={H_value})": policy_balance_future,
            }
    else:
        # Four strategies for comparison
        comparison_strategies = {
            "FCFS": policy_FCFS,
            "Join Shortest Queue": policy_join_shortest_queue,
            "BF-IO(H=0)": policy_balance_future_h0,
            "BF-IO(H=20)": policy_balance_future,
        }

    seeds = np.arange(n_repeat) + global_seed
    eff_variants = ["ratio"]

    for eff_name in eff_variants:
        print(f"\n### Efficiency = {eff_name} ###")
        if H_value is not None:
            print(f"### H = {H_value} ###")
        
        # 1) Generate visualization samples for strategies and plot in one figure
        if not skip_plot:
            print("Generating strategy visualization comparison plots...")
            strategies_sims = {}
            for strat_name, policy_fn in comparison_strategies.items():
                print(f"  Running {strat_name} strategy...")
                # If H value is specified and strategy is BF-IO related, need to run in override_horizon context
                if H_value is not None and args.strategy in ["BF-IO", "BF-IO-H0"]:
                    with override_horizon(H_value):
                        sim_vis, metrics_vis = run_sim_and_metrics(
                            policy_fn,
                            eff_name,
                            N,
                            seed=int(seeds[0]),
                            delta_lower=delta_lower,
                            delta_upper=delta_upper,
                            load_from_real_dataset=load_from_real_dataset,
                            dataset_path=dataset_path,
                            n_worker=n_worker,
                            batch_size=batch_size,
                        )
                else:
                    sim_vis, metrics_vis = run_sim_and_metrics(
                        policy_fn,
                        eff_name,
                        N,
                        seed=int(seeds[0]),
                        delta_lower=delta_lower,
                        delta_upper=delta_upper,
                        load_from_real_dataset=load_from_real_dataset,
                        dataset_path=dataset_path,
                        n_worker=n_worker,
                        batch_size=batch_size,
                    )
                # Store tuple of sim and metrics
                strategies_sims[strat_name] = (sim_vis, metrics_vis)
                
                # Output metrics to console
                print(f"    Avg Imbalance: {metrics_vis.get('avg_gpu_imbalance', 0):.2f}")
                print(f"    Throughput: {metrics_vis.get('throughput_tokens_per_sec', 0):.2f} tok/s")
                print(f"    TPOT: {metrics_vis.get('avg_latency_per_token', 0):.4f} s/tok")
                print(f"    GPU Idle Rate: {metrics_vis.get('avg_gpu_idle_rate', 0):.4f}")
                print(f"    Avg Power: {metrics_vis.get('avg_power', 0):.1f} W")
                print(f"    Total Energy: {metrics_vis.get('total_energy', 0):.2f} J")
            
            plot_four_strategies_comparison(strategies_sims, eff_name=eff_name)
        
        # 2) Parallel simulation + statistical metrics (for all strategies)
        for strat_name, policy_fn in comparison_strategies.items():
            print(f"\nRunning statistical experiments for {strat_name} strategy...")
            # If H value is specified and strategy is BF-IO related (or default BF-IO when no strategy specified), need to run in override_horizon context
            use_horizon = H_value is not None and (
                args.strategy in ["BF-IO", "BF-IO-H0"] if args.strategy else True
            )
            if use_horizon:
                def run_with_h(policy_fn, eff_name, N, seed, delta_lower, delta_upper, 
                              load_from_real_dataset, dataset_path, n_worker, batch_size):
                    with override_horizon(H_value):
                        return run_sim_and_metrics(
                            policy_fn,
                            eff_name,
                            N,
                            seed,
                            delta_lower,
                            delta_upper,
                            load_from_real_dataset,
                            dataset_path,
                            n_worker,
                            batch_size,
                        )
                
                results = Parallel(n_jobs=n_jobs)(
                    delayed(run_with_h)(
                        policy_fn,
                        eff_name,
                        N,
                        int(seed),
                        delta_lower,
                        delta_upper,
                        load_from_real_dataset,
                        dataset_path,
                        n_worker,
                        batch_size,
                    )
                    for seed in tqdm(seeds, desc=f"{strat_name}", ncols=80)
                )
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(run_sim_and_metrics)(
                        policy_fn,
                        eff_name,
                        N,
                        int(seed),
                        delta_lower,
                        delta_upper,
                        load_from_real_dataset,
                        dataset_path,
                        n_worker,
                        batch_size,
                    )
                    for seed in tqdm(seeds, desc=f"{strat_name}", ncols=80)
                )
            # Extract instantaneous power data (from first result, as time series should be similar for all repetitions)
            if results and len(results) > 0:
                _, first_metrics = results[0]
                step_clock_times = first_metrics.get('step_clock_times', [])
                step_powers = first_metrics.get('step_powers', [])
            else:
                step_clock_times = []
                step_powers = []
            
            # Exclude list-type fields (step_clock_times and step_powers) when creating DataFrame
            metrics_list = []
            for _, m in results:
                # Copy metrics and remove list fields
                m_clean = {k: v for k, v in m.items() if k not in ['step_clock_times', 'step_powers']}
                metrics_list.append(m_clean)
            df = pd.DataFrame(metrics_list)

            # Output summary results
            print("\n" + "=" * 60)
            title = f"{strat_name} Strategy (eff = {eff_name})"
            print(title.center(60))
            print("=" * 60)
            for k in df.columns:
                mean = df[k].mean()
                std = df[k].std()
                print(f"{k:>30s}: {mean:.4f} ± {std:.4f}")
            
            # If H value is specified, output key metrics to CSV
            if H_value is not None:
                result_dir = ROOT / "results"
                result_dir.mkdir(parents=True, exist_ok=True)
                csv_file = result_dir / "multiH_results.csv"
                
                # Extract key metrics
                avg_imbalance = df['avg_gpu_imbalance'].mean()
                throughput = df['throughput_tokens_per_sec'].mean()
                tpot = df['avg_latency_per_token'].mean()
                idle_rate = df['avg_gpu_idle_rate'].mean()
                energy = df['total_energy'].mean()
                power = df['avg_power'].mean()
                
                # Write to CSV (append mode)
                import csv
                file_exists = csv_file.exists()
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['H', 'Avg_Imbalance', 'Throughput', 'TPOT', 'Idle_Rate', 'Avg_Power', 'Energy'])
                    writer.writerow([H_value, f"{avg_imbalance:.4f}", f"{throughput:.4f}", f"{tpot:.6f}", f"{idle_rate:.6f}", f"{power:.2f}", f"{energy:.2f}"])
                
                # Save power information to power_<strategy_name>.csv
                strategy_name = f"BF-IO(H={H_value})" if H_value is not None else "BF-IO"
                strategy_file_name = normalize_strategy_name(strategy_name)
                power_csv_file = result_dir / f"power_{strategy_file_name}.csv"
                power_file_exists = power_csv_file.exists()
                with open(power_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not power_file_exists:
                        writer.writerow(['Strategy', 'H', 'B', 'G', 'Avg_Power', 'Energy'])
                    writer.writerow([strategy_name, H_value, batch_size, n_worker, f"{power:.2f}", f"{energy:.2f}"])
                
                # Save instantaneous power to power_timeseries_<strategy_name>.csv
                power_ts_csv_file = result_dir / f"power_timeseries_{strategy_file_name}.csv"
                power_ts_file_exists = power_ts_csv_file.exists()
                with open(power_ts_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not power_ts_file_exists:
                        writer.writerow(['Strategy', 'H', 'B', 'G', 'Time', 'Power'])
                    # Write instantaneous power for each time point
                    for t, p in zip(step_clock_times, step_powers):
                        writer.writerow([strategy_name, H_value, batch_size, n_worker, f"{t:.6f}", f"{p:.2f}"])
                
                print(f"\nResults saved to: {csv_file}")
                print(f"Power information saved to: {power_csv_file}")
                print(f"Instantaneous power time series saved to: {power_ts_csv_file}")
                print(f"H={H_value}: Imbalance={avg_imbalance:.2f}, Throughput={throughput:.2f} tok/s, TPOT={tpot:.4f} s/tok, Idle Rate={idle_rate:.4f}, Avg Power={power:.1f} W, Energy={energy:.2f} J")
            
            # If B value is specified, output key metrics to CSV (when H value or strategy is specified, indicates testing B)
            # Or when B is not default value, also indicates testing B
            if H_value is not None or args.strategy is not None or args.B != 72:  # If H value or strategy is specified (indicates testing B or G), or B is not default
                result_dir = ROOT / "results"
                result_dir.mkdir(parents=True, exist_ok=True)
                
                # Select CSV filename based on strategy type
                if args.strategy == "FCFS":
                    csv_file = result_dir / "multiB_FCFS_results.csv"
                elif args.strategy is not None:
                    csv_file = result_dir / f"multiB_{args.strategy}_results.csv"
                else:
                    csv_file = result_dir / "multiB_results.csv"
                
                # Extract key metrics
                avg_imbalance = df['avg_gpu_imbalance'].mean()
                throughput = df['throughput_tokens_per_sec'].mean()
                tpot = df['avg_latency_per_token'].mean()
                idle_rate = df['avg_gpu_idle_rate'].mean()
                energy = df['total_energy'].mean()
                power = df['avg_power'].mean()
                
                # Write to CSV (append mode)
                import csv
                file_exists = csv_file.exists()
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['B', 'Avg_Imbalance', 'Throughput', 'TPOT', 'Idle_Rate', 'Avg_Power', 'Energy'])
                    writer.writerow([batch_size, f"{avg_imbalance:.4f}", f"{throughput:.4f}", f"{tpot:.6f}", f"{idle_rate:.6f}", f"{power:.2f}", f"{energy:.2f}"])
                
                # Save power information to power_<strategy_name>.csv
                strategy_name = args.strategy if args.strategy else f"BF-IO(H={H_value})" if H_value is not None else "BF-IO"
                strategy_file_name = normalize_strategy_name(strategy_name)
                power_csv_file = result_dir / f"power_{strategy_file_name}.csv"
                power_file_exists = power_csv_file.exists()
                with open(power_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not power_file_exists:
                        writer.writerow(['Strategy', 'H', 'B', 'G', 'Avg_Power', 'Energy'])
                    writer.writerow([strategy_name, H_value if H_value is not None else "", batch_size, n_worker, f"{power:.2f}", f"{energy:.2f}"])
                
                # Save instantaneous power to power_timeseries_<strategy_name>.csv
                power_ts_csv_file = result_dir / f"power_timeseries_{strategy_file_name}.csv"
                power_ts_file_exists = power_ts_csv_file.exists()
                with open(power_ts_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not power_ts_file_exists:
                        writer.writerow(['Strategy', 'H', 'B', 'G', 'Time', 'Power'])
                    # Write instantaneous power for each time point
                    for t, p in zip(step_clock_times, step_powers):
                        writer.writerow([strategy_name, H_value if H_value is not None else "", batch_size, n_worker, f"{t:.6f}", f"{p:.2f}"])
                
                print(f"\nResults saved to: {csv_file}")
                print(f"Power information saved to: {power_csv_file}")
                print(f"Instantaneous power time series saved to: {power_ts_csv_file}")
                print(f"B={batch_size}: Imbalance={avg_imbalance:.2f}, Throughput={throughput:.2f} tok/s, TPOT={tpot:.4f} s/tok, Idle Rate={idle_rate:.4f}, Avg Power={power:.1f} W, Energy={energy:.2f} J")
            
            # If G value is specified, output key metrics to CSV
            # When H value or strategy is specified, indicates testing B or G, need to save results
            # Or when G is not default value, also indicates testing G
            if H_value is not None or args.strategy is not None or args.G != 16:  # If H value or strategy is specified (indicates testing B or G), or G is not default
                result_dir = ROOT / "results"
                result_dir.mkdir(parents=True, exist_ok=True)
                
                # Select CSV filename based on strategy type
                if args.strategy == "FCFS":
                    csv_file = result_dir / "multiG_FCFS_results.csv"
                elif args.strategy is not None:
                    csv_file = result_dir / f"multiG_{args.strategy}_results.csv"
                else:
                    csv_file = result_dir / "multiG_results.csv"
                
                # Extract key metrics
                avg_imbalance = df['avg_gpu_imbalance'].mean()
                throughput = df['throughput_tokens_per_sec'].mean()
                tpot = df['avg_latency_per_token'].mean()
                idle_rate = df['avg_gpu_idle_rate'].mean()
                energy = df['total_energy'].mean()
                power = df['avg_power'].mean()
                
                # Write to CSV (append mode)
                import csv
                file_exists = csv_file.exists()
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['G', 'Avg_Imbalance', 'Throughput', 'TPOT', 'Idle_Rate', 'Avg_Power', 'Energy'])
                    writer.writerow([n_worker, f"{avg_imbalance:.4f}", f"{throughput:.4f}", f"{tpot:.6f}", f"{idle_rate:.6f}", f"{power:.2f}", f"{energy:.2f}"])
                
                # Save power information to power_<strategy_name>.csv
                strategy_name = args.strategy if args.strategy else f"BF-IO(H={H_value})" if H_value is not None else "BF-IO"
                strategy_file_name = normalize_strategy_name(strategy_name)
                power_csv_file = result_dir / f"power_{strategy_file_name}.csv"
                power_file_exists = power_csv_file.exists()
                with open(power_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not power_file_exists:
                        writer.writerow(['Strategy', 'H', 'B', 'G', 'Avg_Power', 'Energy'])
                    writer.writerow([strategy_name, H_value if H_value is not None else "", batch_size, n_worker, f"{power:.2f}", f"{energy:.2f}"])
                
                # Save instantaneous power to power_timeseries_<strategy_name>.csv
                power_ts_csv_file = result_dir / f"power_timeseries_{strategy_file_name}.csv"
                power_ts_file_exists = power_ts_csv_file.exists()
                with open(power_ts_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not power_ts_file_exists:
                        writer.writerow(['Strategy', 'H', 'B', 'G', 'Time', 'Power'])
                    # Write instantaneous power for each time point
                    for t, p in zip(step_clock_times, step_powers):
                        writer.writerow([strategy_name, H_value if H_value is not None else "", batch_size, n_worker, f"{t:.6f}", f"{p:.2f}"])
                
                print(f"\nResults saved to: {csv_file}")
                print(f"Power information saved to: {power_csv_file}")
                print(f"Instantaneous power time series saved to: {power_ts_csv_file}")
                print(f"G={n_worker}: Imbalance={avg_imbalance:.2f}, Throughput={throughput:.2f} tok/s, TPOT={tpot:.4f} s/tok, Idle Rate={idle_rate:.4f}, Avg Power={power:.1f} W, Energy={energy:.2f} J")
