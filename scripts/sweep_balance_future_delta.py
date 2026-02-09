from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simlib import SchedulerSim, generate_requests
from simlib.strategies.strategy_balance_future import policy_balance_future


def run_single_simulation(
    N: int,
    seed: int,
    delta_lower: float,
    delta_upper: float,
    eff_name: str,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    s_arr, o_arr, o_est_arr = generate_requests(
        N,
        rng=rng,
        delta_lower=delta_lower,
        delta_upper=delta_upper,
    )
    sim = SchedulerSim(
        n_worker=32,
        batch_size=72,
        C=(72 * 0.6e-3 + 35e-3) / 8,
        t_length=1.5 * 2 * 576 * 61e-6 / 2**20,
        s_list=s_arr,
        o_list=o_arr,
        o_est_list=o_est_arr,
        policy_fn=policy_balance_future,
        eff_name=eff_name,
        record_history=True,
    )
    return sim.run(verbose=False)


def aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    df = pd.DataFrame(metrics)
    means = df.mean()
    return {f"{k}_mean": float(means[k]) for k in df.columns}


def main() -> None:
    N = 16000
    n_repeat = 100
    global_seed = 42
    eff_name = "ratio"
    output_csv = Path("results") / "balance_future_delta_sweep.csv"
    n_jobs = -1

    delta_values = np.linspace(0.0, 0.1, 21)
    seeds = (np.arange(n_repeat) + global_seed).astype(int)

    results: List[Dict[str, float]] = []
    delta_grid = list(product(delta_values, repeat=2))

    with tqdm(total=len(delta_grid) * n_repeat, desc="Simulations", ncols=80) as sim_pbar:
        for delta_lower, delta_upper in tqdm(delta_grid, desc="Delta grid", ncols=80):
            metrics_per_seed = Parallel(n_jobs=n_jobs)(
                delayed(run_single_simulation)(
                    N=N,
                    seed=int(seed),
                    delta_lower=float(delta_lower),
                    delta_upper=float(delta_upper),
                    eff_name=eff_name,
                )
                for seed in seeds
            )
            sim_pbar.update(n_repeat)

            avg_metrics = aggregate_metrics(list(metrics_per_seed))
            avg_metrics.update(
                {
                    "delta_lower": float(delta_lower),
                    "delta_upper": float(delta_upper),
                }
            )
            results.append(avg_metrics)
            tqdm.write(
                f"Completed delta_lower={delta_lower:.4f}, delta_upper={delta_upper:.4f}"
            )

    df_results = pd.DataFrame(results)
    column_order = ["delta_lower", "delta_upper"] + sorted(
        [col for col in df_results.columns if col not in {"delta_lower", "delta_upper"}]
    )
    df_results = df_results[column_order]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
