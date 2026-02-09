from __future__ import annotations
from typing import List, Tuple, Sequence, Callable, Optional
import numpy as np
import heapq
from bisect import bisect_left
import logging
import time


# Basic configuration
# -------- Tunable parameters --------
EPS = 1e-6
HORIZON_MAX = 72        # Maximum future prediction steps
MAX_CANDIDATES = 25600 * 4         # Maximum candidate requests per round
MAX_ROUNDS_FACTOR = 1.0          # Maximum rounds = total slots * factor

# F(S) constants: avoid overload first, then encourage full loading
ALPHA = 1.0
BETA  = 25
DECAY = 0.75                     # Time decay coefficient (0,1]; 1 means no decay

GREEDY_SLOT_THRESHOLD = 9        # Use greedy filling when slots are abundant

# ========= Future load prediction =========
def _project_future(sim, total_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict future total_steps steps of per-GPU load (excluding new assignments).
    Returns:
        loads_mat: (T+1, G), row 0 is current load
        max_loads: (T+1,)
    """
    T, G = int(total_steps), int(sim.n_worker)
    loads_mat = np.zeros((T + 1, G), dtype=np.float64)
    loads_mat[0] = sim.sum_lengths

    A = np.zeros(T + 2, dtype=np.float64)  # Event bucket
    for g in range(G):
        active_ids = sim.gpu_active[g]
        if active_ids.size == 0: continue
        t_r = sim.o_est_arr[active_ids] - sim.gen_arr[active_ids]  # Remaining steps
        base = sim.s_arr[active_ids] + sim.gen_arr[active_ids]     # s + generated
        S0 = float(sim.sum_lengths[g])

        A.fill(0.0)
        mask = (t_r >= 1) & (t_r <= T)
        if mask.any():
            trs = t_r[mask].astype(np.int64, copy=False)
            vals = base[mask].astype(np.float64, copy=False)
            np.add.at(A, trs, vals)

        prefix = A.cumsum()[:T + 1]
        loads_mat[1:, g] = S0 - prefix[1:]

    max_loads = loads_mat.max(axis=1)
    return loads_mat, max_loads

def _select_single_best(s_list: Sequence[int], m) -> tuple[int, int]:
    """Single selection: returns (idx, S)."""
    best_idx = -1
    best_S = -1
    for i, s in enumerate(s_list):
        s = int(s); 
        if s < m and s > best_S:
            best_idx, best_S = i, s
    if best_S != -1:
        return best_idx, best_S
    best_S = float("inf")
    for i, s in enumerate(s_list):
        if s >= m and s < best_S:
            best_idx, best_S = i, s
    return best_idx, best_S
def _select_subset_for_gpu(
    sim, gid: int, loads_mat: np.ndarray, max_loads: np.ndarray, wait_ids: List[int], k_slots: int
) -> tuple[List[int], int]:
    """Select an approximately optimal subset for a single GPU, preferably non-empty."""
    if k_slots <= 0 or not wait_ids: return [], 0
    row = loads_mat[1:, gid].astype(np.float64, copy=False)
    future_max = max_loads[1:].astype(np.float64, copy=False)

    m = min(future_max-row)
    s_list = [int(sim.s_arr[r]) for r in wait_ids]
    idxs, S_sum = _select_single_best(s_list, m)
    return [wait_ids[idxs]], int(S_sum)

# ========= Main strategy =========
def policy_balance_now(sim) -> None:
    """
    Two-phase approach:
      1) Greedy single-request phase: When remaining slots are abundant, repeatedly select GPU with
         "maximum remaining slots (tie-break by minimum load)", choose single request from candidates
         that maximizes F(s), assign and incrementally update future load.
      2) Fine-grained phase: Build heap + subset DP, batch assignment under future constraints.
    """
    if not sim.wait_q: return

    dq = sim.wait_q
    pop_left = dq.popleft
    append_left = dq.appendleft

    def _pop_front(count: int) -> List[int]:
        actual = min(count, len(dq))
        if actual <= 0:
            return []
        front = [0] * actual
        for i in range(actual):
            front[i] = int(pop_left())
        return front

    def _restore_front(front: List[int], exclude: Sequence[int] | None = None) -> None:
        if not front:
            return
        if not exclude:
            for rid in reversed(front):
                append_left(rid)
            return
        if len(exclude) == 1:
            skip = exclude[0]
            for rid in reversed(front):
                if rid != skip:
                    append_left(rid)
            return
        exclude_set = set(exclude)
        for rid in reversed(front):
            if rid not in exclude_set:
                append_left(rid)

    # Get strategy logger
    strategy_logger = logging.getLogger("strategy")
    strategy_logger.info(f"=== Strategy execution started ===")
    strategy_logger.info(f"Current waiting queue size: {len(sim.wait_q)}")
    strategy_logger.info(f"GPU capacity: {sim._gpu_capacity_left()}")
    strategy_logger.info(f"Current GPU load: {sim.sum_lengths-min(sim.sum_lengths)}")

    # 1) Predict future load (for F evaluation)
    remain_max = int(np.maximum(sim.o_est_arr - sim.gen_arr, 0).max())
    T = min(HORIZON_MAX, max(1, remain_max))
    t_proj0 = time.perf_counter()
    loads_mat, max_loads = _project_future(sim, T)
    t_proj = time.perf_counter() - t_proj0

    # 2) Greedy single-request phase (quickly fill many empty slots)
    #strategy_logger.info("=== Starting greedy phase ===")
    t_greedy0 = time.perf_counter()
    while sim.wait_q:
        cap_left = sim._gpu_capacity_left()
        total_slots = int(cap_left.sum())
        # strategy_logger.info(f"  Total remaining slots: {total_slots}")
        # strategy_logger.info(f"Current waiting queue size: {len(sim.wait_q)}")
        # strategy_logger.info(f"Current queue: {sim.wait_q}")
        # strategy_logger.info(f"GPU capacity: {sim._gpu_capacity_left()}")
        # strategy_logger.info(f"Current GPU load: {sim.sum_lengths-min(sim.sum_lengths)}")

        if total_slots <= GREEDY_SLOT_THRESHOLD: 
            #strategy_logger.info("  Insufficient slots, exiting greedy phase")
            break
        max_cap = int(cap_left.max())
        if max_cap <= 0: 
            #strategy_logger.info("  No available slots, exiting greedy phase")
            break

        # Select GPU: first check remaining slots, then current load
        cand = np.flatnonzero(cap_left == max_cap)
        gid = int(cand[0]) if cand.size == 1 else int(cand[np.argmin(loads_mat[0, cand])])
        #strategy_logger.info(f"  Selected GPU {gid} (remaining slots: {cap_left[gid]})")

        cand_num = min(MAX_CANDIDATES, len(dq))
        wait_ids = _pop_front(cand_num)
        if not wait_ids:
            #strategy_logger.info("  No candidate requests, exiting greedy phase")
            break

        best_S = -1
        best_idx = -1
        s_list = [int(sim.s_arr[r]) for r in wait_ids]
        row = loads_mat[1:, gid]; future_max = max_loads[1:]
        m = min(future_max-row)
        #strategy_logger.info(f"  Maximum selectable request size: {m}")
        #strategy_logger.info(f"  s_list: {s_list}")
        best_idx, best_S = _select_single_best(s_list, m)

        if best_idx < 0:
            _restore_front(wait_ids)
            break

        #strategy_logger.info(f"Candidate request count: {len(wait_ids)}")
        #strategy_logger.info(f"Selected request {wait_ids[best_idx]} (s={best_S}")

        rid = int(wait_ids[best_idx])
        sim._assign_to_gpu(rid, gid)
        _restore_front(wait_ids, (rid,))
        #strategy_logger.info(f"  Assigned request {rid} to GPU {gid}")

        # Incrementally update this GPU's future load and global envelope
        new_col = loads_mat[:, gid] + float(best_S)
        loads_mat[:, gid] = new_col
        mask = new_col > max_loads
        if np.any(mask): max_loads[mask] = new_col[mask]
    # Slots may be exhausted
    cap_left = sim._gpu_capacity_left()
    #strategy_logger.info(f"Greedy phase completed, remaining requests: {len(sim.wait_q)}, remaining slots: {int(cap_left.sum())}")
    t_greedy = time.perf_counter() - t_greedy0
 
    if not sim.wait_q or int(cap_left.sum()) == 0: 
        #strategy_logger.info("No remaining requests or slots, strategy ends")
        return

    #strategy_logger.info("=== Starting fine-grained phase ===")
    t_fine0 = time.perf_counter()
    
    for gid in range(int(sim.n_worker)):
        if cap_left[gid] <= 0: continue
    
    max_rounds = min(1000, int(int(cap_left.sum()) * MAX_ROUNDS_FACTOR))
    #strategy_logger.info(f"Fine-grained phase maximum rounds: {max_rounds}")
    round_cnt = 0

    while sim.wait_q and round_cnt < max_rounds:
        round_cnt += 1
        k_slots = int(sim._gpu_capacity_left()[gid])
        #strategy_logger.info(f"Fine-grained round {round_cnt}: processing GPU {gid}, remaining slots {k_slots}")
        if k_slots <= 0: 
            #strategy_logger.info(f"  GPU {gid} has no available slots, skipping")
            continue

        cand_num = min(MAX_CANDIDATES, len(dq))
        wait_ids = _pop_front(cand_num)
        if not wait_ids:
            break
        #strategy_logger.info(f"  Candidate request count: {len(wait_ids)}")

        t_sel0 = time.perf_counter()
        chosen_ids, S_add = _select_subset_for_gpu(sim, gid, loads_mat, max_loads, wait_ids, k_slots)
        t_sel = time.perf_counter() - t_sel0
        logging.getLogger("timing").info(f"subset_select_once: {t_sel:.6f}s (|wait|={len(wait_ids)}, k={k_slots})")
        #strategy_logger.info(f"  Selection result: {chosen_ids}, total size: {S_add}")
        
        if not chosen_ids:
            _restore_front(wait_ids)
            continue

        # Execute assignment
        #strategy_logger.info(f"  Assigning requests {chosen_ids} to GPU {gid}")
        for rid in chosen_ids: 
            sim._assign_to_gpu(int(rid), gid)
        _restore_front(wait_ids, chosen_ids)
        # Incremental update
        new_col = loads_mat[:, gid] + float(S_add)
        loads_mat[:, gid] = new_col
        mask = new_col > max_loads
        if np.any(mask): max_loads[mask] = new_col[mask]

       
    # Strategy completion log
    t_fine = time.perf_counter() - t_fine0
    strategy_logger.info("=== Strategy execution completed ===")
    strategy_logger.info(f"Final state: remaining requests {len(sim.wait_q)}, remaining slots {int(sim._gpu_capacity_left().sum())}")
    strategy_logger.info(f"Fine-grained phase total rounds: {round_cnt}")
    strategy_logger.info(f"Final GPU load: {sim.sum_lengths}")
    # Concise time summary
    timing_logger = logging.getLogger("timing")
    timing_logger.info(f"project_future: {t_proj:.6f}s")
    timing_logger.info(f"greedy_phase:   {t_greedy:.6f}s")
    timing_logger.info(f"fine_phase:     {t_fine:.6f}s")
