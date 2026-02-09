# milp_token_balancer.py
"""
Solve a MILP to plan token-level GPU scheduling over T steps to minimize average imbalance:
    imbalance[t] = n_worker * max_load[t] - sum_load[t]

Assumptions:
- Each GPU has limited concurrent request capacity.
- Each request has input size s_i and output length o_i (token budget).
- Each request once scheduled occupies a GPU until all tokens are generated (non-preemptive).
- At each step, each active request generates 1 token.

Requirements:
    pip install cvxpy numpy
"""

import sys
from pathlib import Path
from typing import Dict, Any

import cvxpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simlib import generate_requests


def solve_token_balancing(
    s_list: np.ndarray,
    o_list: np.ndarray,
    G: int = 4,
    T: int = 100,
    batch_size: int = 72,
    solver_preference=("GUROBI", "CBC", "GLPK_MI", "SCIP", "ECOS_BB"),
) -> Dict[str, Any]:
    """
    MILP solver for scheduling requests to GPUs over T steps to minimize average imbalance.
    
    Args:
        s_list: array of input lengths (s_i)
        o_list: array of output token goals (o_i)
        G: number of GPUs
        T: number of discrete decode steps
        batch_size: maximum concurrent requests per GPU

    Returns:
        Dict with imbalance, GPU loads, assignment schedule.
    """
    N = len(s_list)
    s = np.asarray(s_list, dtype=int)
    o = np.asarray(o_list, dtype=int)
    total_token = np.sum(o)

    # === Variables ===
    start = cp.Variable((N, T), boolean=True)        # start[i,t] = 1 if request i starts at step t
    assign = cp.Variable((N, G), boolean=True)       # assign[i,g] = 1 if request i is assigned to GPU g
    active = cp.Variable((N, G, T), boolean=True)    # active[i,g,t] = 1 if request i is on g at time t

    max_load = cp.Variable(T)                        # max GPU load at time t
    imbalance = cp.Variable(T)                       # n*max - sum

    constraints = []

    # (1) Each request starts once
    constraints.append(cp.sum(start, axis=1) == 1)

    # (2) Each request assigned to one GPU
    constraints.append(cp.sum(assign, axis=1) == 1)

    # (3) Active[i,g,t] = 1 iff assigned[i,g] and started at t0 <= t < t0+o[i]
    for i in range(N):
        for g in range(G):
            for t in range(T):
                act_expr = []
                for tau in range(max(0, t - o[i] + 1), t + 1):
                    act_expr.append(start[i, tau])
                if act_expr:
                    constraints.append(active[i, g, t] <= assign[i, g])
                    constraints.append(active[i, g, t] <= cp.sum(act_expr))
                    constraints.append(active[i, g, t] >= assign[i, g] + cp.sum(act_expr) - 1)
                else:
                    constraints.append(active[i, g, t] == 0)

    # (4) GPU capacity per step
    for g in range(G):
        for t in range(T):
            constraints.append(cp.sum(active[:, g, t]) <= batch_size)

    # (5) Define GPU load per step
    load = []  # load[g,t]
    for t in range(T):
        gpu_load_t = []
        for g in range(G):
            l_gt = cp.sum(cp.multiply(active[:, g, t], s + 1))  # s_i + 1 token per active
            gpu_load_t.append(l_gt)
            constraints.append(max_load[t] >= l_gt)
        load.append(gpu_load_t)
        constraints.append(imbalance[t] == G * max_load[t] - cp.sum(gpu_load_t))

    # (6) 每个请求必须完整生成 o_i 个 token（活跃 o_i 步）
    for i in range(N):
        total_active = cp.sum(active[i, :, :])
        constraints.append(total_active == o[i])

    # === Objective ===
    obj = cp.Minimize(cp.sum(imbalance) / T)
    prob = cp.Problem(obj, constraints)

    # === Solve ===
    chosen = next((s for s in solver_preference if s in cp.installed_solvers()), None)
    prob.solve(solver=chosen)

    return dict(
        status=prob.status,
        avg_imbalance=prob.value,
        max_load=np.round([max_l.value for max_l in max_load], 2),
        assignment=np.round(assign.value, 3),
        start=np.round(start.value, 3),
        active=np.round(active.value, 3),
    )

# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    s_list, o_list = generate_requests(100)  # 1000 requests
    result = solve_token_balancing(s_list, o_list, G=4, T=100, batch_size=72)

    print("Status:", result['status'])
    print("Average imbalance:", result['avg_imbalance'])
