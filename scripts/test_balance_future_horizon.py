from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simlib import SchedulerSim, generate_requests
from simlib.strategies import strategy_balance_future as balance_future_mod


# ------------------- 全局配置 -------------------
N_REQUESTS = 8000
N_REPEAT = 100
N_JOBS = -1
GLOBAL_SEED = 42
DELTA_LOWER = 0.0
DELTA_UPPER = 0.0
H_VALUES = [0,1,2,4,8,16,32,64,128,256,512]#list(range(0, 129, 16))
MAX_PLOTTED_STEPS = 8000
SMOOTH_WINDOW = 500

RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def override_horizon(h: int):
    """临时修改策略中的 HORIZON_MAX 常量。"""
    original = balance_future_mod.HORIZON_MAX
    balance_future_mod.HORIZON_MAX = int(h)
    try:
        yield
    finally:
        balance_future_mod.HORIZON_MAX = original


def build_simulator(
    s_arr: np.ndarray,
    o_arr: np.ndarray,
    o_est_arr: np.ndarray,
) -> SchedulerSim:
    return SchedulerSim(
        n_worker=16,
        batch_size=72,
        C=(72 * 0.6e-3 + 35e-3) / 8,
        t_length=1.5 * 2 * 576 * 61e-6 / 2**20,
        s_list=s_arr,
        o_list=o_arr,
        o_est_list=o_est_arr,
        policy_fn=balance_future_mod.policy_balance_future,
        eff_name="ratio",
        record_history=True,
    )


def run_sim_with_horizon(
    horizon: int,
    s_arr: np.ndarray,
    o_arr: np.ndarray,
    o_est_arr: np.ndarray,
) -> SchedulerSim:
    with override_horizon(horizon):
        sim = build_simulator(s_arr, o_arr, o_est_arr)
        sim.run(verbose=False)
    return sim


def compute_imbalance_series(sim: SchedulerSim, max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    limit = min(max_steps, len(sim.hist_sum_lengths))
    if limit == 0:
        return np.array([]), np.array([])
    loads = np.vstack(sim.hist_sum_lengths[:limit])
    steps = np.array(sim.hist_steps[:limit])
    max_loads = loads.max(axis=1)
    sum_loads = loads.sum(axis=1)
    imbalance = sim.n_worker * max_loads - sum_loads
    return steps, imbalance


def compute_avg_imbalance(sim: SchedulerSim, max_steps: int) -> float:
    _, imbalance = compute_imbalance_series(sim, max_steps)
    return float(imbalance.mean()) if imbalance.size else 0.0


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def simulate_avg_imbalance(horizon: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    s_arr, o_arr, o_est_arr = generate_requests(
        N_REQUESTS,
        rng=rng,
        delta_lower=DELTA_LOWER,
        delta_upper=DELTA_UPPER,
    )
    sim = run_sim_with_horizon(horizon, s_arr, o_arr, o_est_arr)
    return compute_avg_imbalance(sim, MAX_PLOTTED_STEPS)


def plot_imbalance_over_time(series: Dict[int, Tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(series)))
    for color, (h, (steps, imbs)) in zip(colors, series.items()):
        if steps.size == 0:
            continue
        ax.plot(
            steps,
            imbs,
            label=f"H = {h}",
            linewidth=2.6,
            color=color,
            alpha=0.92,
        )
    ax.set_xlabel("Decode step", fontsize=21, fontweight="semibold", color="#1f1f2e")
    ax.set_ylabel("Imbalance", fontsize=21, fontweight="semibold", color="#1f1f2e")
    ax.tick_params(axis="both", labelsize=18, colors="#2b2b3c", width=2.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val / 1000:.0f}k"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35, color="#7c8aa6")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2.2)
        ax.spines[spine].set_color("#1f1f2e")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("#f4f6fb")
    fig.patch.set_facecolor("#eef1f7")
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
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=600, format='jpg')
    plt.close(fig)


def plot_smoothed_imbalance_over_time(
    series: Dict[int, Tuple[np.ndarray, np.ndarray]],
    window: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(series)))
    for color, (h, (steps, imbs)) in zip(colors, series.items()):
        if steps.size == 0:
            continue
        smooth_vals = smooth_series(imbs, window)
        ax.plot(
            steps,
            smooth_vals,
            label=f"H = {h}",
            linewidth=2.8,
            color=color,
            alpha=0.95,
        )
    ax.set_xlabel("Decode step", fontsize=21, fontweight="semibold", color="#1f1f2e")
    ax.set_ylabel(f"Imbalance (smoothed)", fontsize=21, fontweight="semibold", color="#1f1f2e")
    ax.tick_params(axis="both", labelsize=18, colors="#2b2b3c", width=2.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val / 1000:.0f}k"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35, color="#7c8aa6")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2.2)
        ax.spines[spine].set_color("#1f1f2e")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("#f4f6fb")
    fig.patch.set_facecolor("#eef1f7")
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
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=600, format='jpg')
    plt.close(fig)


def plot_avg_vs_h(h_values: Iterable[int], means: np.ndarray, stds: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.0))
    color = "#3366cc"
    ax.plot(
        list(h_values),
        means,
        marker="o",
        linewidth=3.0,
        markersize=9,
        color=color,
        alpha=0.95,
        label="Mean imbalance",
    )
    if stds.size:
        ax.fill_between(
            list(h_values),
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            label="±1 std",
        )
    ax.set_xlabel("Window Length (H)", fontsize=21, fontweight="semibold", color="#1f1f2e")
    ax.set_ylabel("Avg Imbalance", fontsize=20, fontweight="semibold", color="#1f1f2e")
    ax.tick_params(axis="both", labelsize=18, colors="#2b2b3c", width=2.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val / 1000:.0f}k"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35, color="#7c8aa6")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2.2)
        ax.spines[spine].set_color("#1f1f2e")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("#f4f6fb")
    fig.patch.set_facecolor("#eef1f7")
    ax.legend(fontsize=16, frameon=True, fancybox=True, framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=600, format='jpg')
    plt.close(fig)


def main():
    base_rng = np.random.default_rng(GLOBAL_SEED)
    base_s, base_o, base_o_est = generate_requests(
        N_REQUESTS,
        rng=base_rng,
        delta_lower=DELTA_LOWER,
        delta_upper=DELTA_UPPER,
    )

    print("Running single simulations for imbalance trajectories...")
    horizon_series: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for h in H_VALUES:
        sim = run_sim_with_horizon(h, base_s, base_o, base_o_est)
        horizon_series[h] = compute_imbalance_series(sim, MAX_PLOTTED_STEPS)

    plot_imbalance_over_time(
        horizon_series,
        FIG_DIR / "balance_future_imbalance_vs_time.jpg",
    )
    plot_smoothed_imbalance_over_time(
        horizon_series,
        SMOOTH_WINDOW,
        FIG_DIR / "balance_future_imbalance_vs_time_smoothed.jpg",
    )

    print("Running parallel sweep for average imbalance curve...")
    seeds = np.arange(N_REPEAT) + GLOBAL_SEED
    mean_vals = []
    std_vals = []
    for h in tqdm(H_VALUES, desc="H sweep", ncols=85):
        samples = Parallel(n_jobs=N_JOBS)(
            delayed(simulate_avg_imbalance)(int(h), int(seed))
            for seed in seeds
        )
        arr = np.asarray(samples, dtype=np.float64)
        mean_vals.append(arr.mean())
        std_vals.append(arr.std())

    plot_avg_vs_h(
        H_VALUES,
        np.asarray(mean_vals),
        np.asarray(std_vals),
        FIG_DIR / "balance_future_avg_imbalance_vs_H.jpg",
    )


if __name__ == "__main__":
    main()
