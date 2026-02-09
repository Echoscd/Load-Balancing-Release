from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simlib.strategies.strategy_balance_future as st
from simlib import SchedulerSim, generate_requests
from simlib.strategies.strategy_balance_future import policy_balance_future

# ------------------- 单次仿真，返回指标 -------------------
def run_one_sim(
    s_arr: np.ndarray,
    o_arr: np.ndarray,
    o_est_arr: np.ndarray,
    eff_name: str = "ratio",
) -> Dict[str, float]:
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
        record_history=False,  # 节省内存
    )
    return sim.run(verbose=False)

# ------------------- 让 objective 捕获一次性读入的数据 -------------------
def make_objective(
    s_arr: np.ndarray,
    o_arr: np.ndarray,
    o_est_arr: np.ndarray,
    eff_name: str = "ratio",
    repeats: int = 1,   # 固定数据建议 1；策略含随机性可增大
):
    def objective(trial: optuna.Trial) -> float:
        # 1) 采样超参并写回策略模块全局变量
        st.HORIZON_MAX           = trial.suggest_int( "HORIZON_MAX",           16, 128, step=8)
        st.BETA                  = trial.suggest_float("BETA",                 1e1, 1e3, log=True)
        st.DECAY                 = trial.suggest_float("DECAY",                0.5, 1.0)
        st.GREEDY_SLOT_THRESHOLD = trial.suggest_int( "GREEDY_SLOT_THRESHOLD", 0, 16)

        # 2) 使用同一份数据重复跑 repeats 次取均值
        vals = []
        for _ in range(repeats):
            m = run_one_sim(s_arr, o_arr, o_est_arr, eff_name=eff_name)
            vals.append(float(m["avg_gpu_imbalance"]))  # 指标键名按实际为准
        return float(np.mean(vals))
    return objective

# ------------------- 主程序 -------------------
if __name__ == "__main__":
    N, global_seed, repeats = 16000, 42, 1
    eff_name = "ratio"

    rng = np.random.default_rng(global_seed)
    s_arr, o_arr, o_est_arr = generate_requests(N, rng=rng)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=global_seed),
    )
    study.optimize(
        make_objective(s_arr, o_arr, o_est_arr, eff_name=eff_name, repeats=repeats),
        n_trials=80,
        show_progress_bar=True,
        n_jobs=1,   # 明确单进程
    )

    # 结果输出
    print("\n" + "=" * 60)
    print("Best Hyper-parameters".center(60))
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"{k:>25s}: {v}")
    print(f"{'Best mean imbalance':>25s}: {study.best_value:.4f}")

    # 收敛曲线
    df = study.trials_dataframe()
    plt.plot(df["value"], marker="o")
    plt.xlabel("Trial"); plt.ylabel("Mean imbalance")
    plt.title("Optuna Convergence"); plt.tight_layout(); plt.show()
