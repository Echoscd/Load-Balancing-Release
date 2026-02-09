from __future__ import annotations
from typing import List, Tuple, Sequence, Callable, Optional
import numpy as np
import heapq
from bisect import bisect_left

# -------- Tunable parameters --------
EPS = 1e-6
HORIZON_MAX = 72                 # Maximum future prediction steps
MAX_CANDIDATES = 256 * 4         # Maximum candidate requests per round
MAX_ROUNDS_FACTOR = 1.0          # Maximum rounds = total slots * factor

# F(S) constants: avoid overload first, then encourage full loading
ALPHA = 1.0
BETA  = 43
DECAY = 0.86                     # Time decay coefficient (0,1]; 1 means no decay

FAIL_FORCE_THRESHOLD = 3         # Consecutive failure threshold: fallback to assign 1 request
GREEDY_SLOT_THRESHOLD = 7        # Use greedy filling when slots are abundant

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

# ========= λ calculation =========
def compute_lambda_from_loads(sim, loads_mat, tol: float = 1e-6) -> float:
    """Use estimated steps for future max load to decay to ~0, balance current vs future."""
    maxv = loads_mat[1:].max(axis=1)  # (T,)
    if maxv.size == 0:
        rem_steps = 0
    else:
        hit = np.nonzero(maxv <= tol)[0]
        rem_steps = 1 if (maxv[0] <= tol) else (int(hit[0] + 1) if hit.size else maxv.size)
    est_remain_time = rem_steps * float(sim.t_length)
    lam = float(sim.clock) / (float(sim.clock) + est_remain_time + EPS)
    return float(np.clip(lam, 0.0, 1.0))

# ========= F(S) evaluation (with time decay; query O(log T)) =========
def build_F_evaluator_fast(
    load_row: np.ndarray,
    max_loads: np.ndarray,
) -> tuple[Callable[[int], float], int]:
    """
    margin_t = max_loads[t] - load_row[t], t=0..T; d_t=DECAY**t
    F(S) = α·S·Σd_t  -  β·[ S·Σ_{m<S} d_t  -  Σ_{m<S}(d_t·m) ]
    Returns: F_of(S), safety upper bound m = floor(min_t margin_t)
    """
    margin = (max_loads - load_row).astype(np.float64, copy=False)
    np.maximum(margin, 0.0, out=margin)
    T = int(margin.size)
    if T == 0:
        return (lambda _S: 0.0), 0

    decay = DECAY ** np.arange(T, dtype=np.float64)
    order = np.argsort(margin, kind="mergesort")
    ms = margin[order]
    ds = decay[order]

    prefix_d  = np.concatenate(([0.0], np.cumsum(ds)))
    prefix_dm = np.concatenate(([0.0], np.cumsum(ds * ms)))
    αD = ALPHA * prefix_d[-1]
    m = int(np.floor(float(ms[0])))

    def F_of(S: int) -> float:
        S = int(S)
        if S <= 0: return 0.0
        idx = bisect_left(ms, float(S))     # First position >= S
        if idx == 0: return αD * S          # All safe
        overflow_d, overflow_dm = prefix_d[idx], prefix_dm[idx]
        return αD * S - BETA * (S * overflow_d - overflow_dm)

    return F_of, m

# ========= Subset selection (bitset DP) =========
def _best_S_from_reach(reach: int, m: int, S_cap: int, F_of: Callable[[int], float]) -> tuple[int, float]:
    """In reach (bitset of reachable sums), prefer maximum S≤m; otherwise take minimum S>m."""
    if reach == 0: return 0, float("-inf")
    if m >= 1:
        r1 = reach & ((1 << (m + 1)) - 1)
        if r1:
            S1 = r1.bit_length() - 1
            return S1, F_of(S1)
    if m + 1 <= S_cap:
        r2 = reach >> (m + 1)
        if r2:
            pos = (r2 & -r2).bit_length() - 1
            S2 = (m + 1) + pos
            return S2, F_of(S2)
    return 0, float("-inf")

def select_subset_fixed_k_fast(
    s_list: Sequence[int], k: int, F_of: Callable[[int], float], S_cap: int, m: int
) -> tuple[float, List[int], int]:
    """Select exactly k items to maximize F(sum); returns (best_F, indices, best_S)."""
    n = len(s_list)
    if k <= 0 or n == 0: return 0.0, [], 0
    k = min(k, n); S_cap = max(1, int(S_cap))

    dp = [0] * (k + 1); dp[0] = 1
    snaps = []
    for i, s in enumerate(s_list):
        s = int(s); snaps.append(dp.copy())
        topj = min(i + 1, k)
        for j in range(topj, 0, -1):
            dp[j] |= (dp[j - 1] << s)
            dp[j] &= (1 << (S_cap + 1)) - 1

    reach = dp[k]
    if reach == 0: return 0.0, [], 0
    best_S, best_F = _best_S_from_reach(reach, m, S_cap, F_of)
    if best_F == float("-inf"): return 0.0, [], 0

    # Backtrack
    chosen: List[int] = []
    curS, curj = best_S, k
    for i in range(n - 1, -1, -1):
        if curj == 0: break
        prev = snaps[i]; s = int(s_list[i])
        if curS >= s and ((prev[curj - 1] >> (curS - s)) & 1):
            chosen.append(i); curS -= s; curj -= 1
    chosen.reverse()
    return (float(best_F), chosen, best_S) if chosen else (0.0, [], 0)

def _try_select_subset_any_k_fast(
    s_list: Sequence[int], k_slots: int, F_of: Callable[[int], float], m: int
) -> tuple[List[int], int, float]:
    """Try k_slots..1 progressively; S_cap bounded quickly using prefix sum."""
    n = len(s_list)
    if n == 0 or k_slots <= 0: return [], 0, float("-inf")

    sorted_s = np.sort(np.asarray(s_list, dtype=np.int64))[::-1]
    prefix = np.cumsum(sorted_s, dtype=np.int64)

    best_choice: List[int] = []; best_S = 0; best_F = float("-inf")
    for k in range(min(k_slots, n), 0, -1):
        S_cap = int(prefix[k - 1])
        cur_F, idxs, cur_S = select_subset_fixed_k_fast(s_list, k, F_of, S_cap, m)
        if idxs and cur_F > best_F:
            best_F, best_S, best_choice = cur_F, cur_S, idxs
            if best_S <= m: break  # Reached maximum S in "safe zone", early exit
    return best_choice, best_S, best_F

def _select_single_best(s_list: Sequence[int], F_of: Callable[[int], float]) -> tuple[int, float, int]:
    """Single selection: returns (idx, F, S)."""
    best_idx = -1; best_F = float("-inf"); best_S = 0
    for i, s in enumerate(s_list):
        s = int(s); val = F_of(s)
        if val > best_F: best_idx, best_F, best_S = i, val, s
    return best_idx, best_F, best_S

def _select_subset_for_gpu(
    sim, gid: int, loads_mat: np.ndarray, max_loads: np.ndarray, wait_ids: List[int], k_slots: int
) -> tuple[List[int], int]:
    """Select an approximately optimal subset for a single GPU, preferably non-empty."""
    if k_slots <= 0 or not wait_ids: return [], 0
    row = loads_mat[:, gid].astype(np.float64, copy=False)
    future_max = max_loads.astype(np.float64, copy=False)
    F_of, m = build_F_evaluator_fast(row, future_max)

    s_list = [int(sim.s_arr[r]) for r in wait_ids]
    idxs, S_sum, best_F = _try_select_subset_any_k_fast(s_list, k_slots, F_of, m)
    if (not idxs) or (best_F <= 0.0):
        idx, _, S1 = _select_single_best(s_list, F_of)
        return ([wait_ids[idx]], int(S1)) if idx >= 0 else ([], 0)
    return [wait_ids[i] for i in idxs], int(S_sum)

# ========= Main strategy =========
def policy_balance_future(sim) -> None:
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

    # 1) Predict future load (for F evaluation)
    remain_max = int(np.maximum(sim.o_est_arr - sim.gen_arr, 0).max())
    T = min(HORIZON_MAX, max(1, remain_max))
    loads_mat, max_loads = _project_future(sim, T)

    # 2) Greedy single-request phase (quickly fill many empty slots)
    while sim.wait_q:
        cap_left = sim._gpu_capacity_left()
        total_slots = int(cap_left.sum())
        if total_slots <= GREEDY_SLOT_THRESHOLD: break
        max_cap = int(cap_left.max())
        if max_cap <= 0: break

        # Select GPU: first check remaining slots, then current load
        cand = np.flatnonzero(cap_left == max_cap)
        gid = int(cand[0]) if cand.size == 1 else int(cand[np.argmin(loads_mat[0, cand])])

        # F(s)
        row = loads_mat[:, gid]; future_max = max_loads
        F_of, _ = build_F_evaluator_fast(row, future_max)

        cand_num = min(MAX_CANDIDATES, len(dq))
        wait_ids = _pop_front(cand_num)
        if not wait_ids:
            break

        # Prefer maximum F in s>0; if no positive s, choose from all
        best_idx = -1; best_F = float("-inf"); best_S = 0
        for i, rid in enumerate(wait_ids):
            s = int(sim.s_arr[rid])
            if s <= 0: continue
            val = F_of(s)
            if val > best_F: best_idx, best_F, best_S = i, val, s
        if best_idx < 0:
            for i, rid in enumerate(wait_ids):
                s = int(sim.s_arr[rid]); val = F_of(s)
                if val > best_F: best_idx, best_F, best_S = i, val, s
        if best_idx < 0:
            _restore_front(wait_ids)
            break

        rid = int(wait_ids[best_idx])
        sim._assign_to_gpu(rid, gid)
        _restore_front(wait_ids, (rid,))

        # Incrementally update this GPU's future load and global envelope
        new_col = loads_mat[:, gid] + float(best_S)
        loads_mat[:, gid] = new_col
        mask = new_col > max_loads
        if np.any(mask): max_loads[mask] = new_col[mask]

    # Slots may be exhausted
    cap_left = sim._gpu_capacity_left()
    if not sim.wait_q or int(cap_left.sum()) == 0: return

    # 3) Fine-grained phase: heap + subset DP
    lam = compute_lambda_from_loads(sim, loads_mat)
    heap_: List[Tuple[float, int]] = []
    for gid in range(int(sim.n_worker)):
        if cap_left[gid] <= 0: continue
        cur, fut = float(loads_mat[0, gid]), float(loads_mat[-1, gid])
        heapq.heappush(heap_, ((1.0 - lam) * cur + lam * fut, gid))
    if not heap_: return

    max_rounds = min(1000, int(int(cap_left.sum()) * MAX_ROUNDS_FACTOR))
    round_cnt = 0
    fail_streak = [0] * int(sim.n_worker)

    while heap_ and sim.wait_q and round_cnt < max_rounds:
        _, gid = heapq.heappop(heap_)
        k_slots = sim.batch_size - len(sim.gpu_active[gid])
        if k_slots <= 0: continue

        cand_num = min(MAX_CANDIDATES, len(dq))
        wait_ids = _pop_front(cand_num)

        chosen_ids, S_add = _select_subset_for_gpu(sim, gid, loads_mat, max_loads, wait_ids, k_slots)

        # Fallback: force select 1 after consecutive failures
        if (not chosen_ids) and fail_streak[gid] + 1 >= FAIL_FORCE_THRESHOLD and wait_ids:
            row = loads_mat[:, gid]; future_max = max_loads
            F_of, _ = build_F_evaluator_fast(row, future_max)
            s_list = [int(sim.s_arr[r]) for r in wait_ids]
            idx, _, S1 = _select_single_best(s_list, F_of)
            if idx >= 0:
                chosen_ids, S_add = [wait_ids[idx]], int(S1)

        if not chosen_ids:
            fail_streak[gid] += 1
            _restore_front(wait_ids)
            cur, fut = float(loads_mat[0, gid]), float(loads_mat[-1, gid])
            heapq.heappush(heap_, ((1.0 - lam) * cur + lam * fut, gid))
            round_cnt += 1
            continue

        # Execute assignment
        for rid in chosen_ids: sim._assign_to_gpu(int(rid), gid)
        _restore_front(wait_ids, chosen_ids)
        fail_streak[gid] = 0

        # Incremental update
        new_col = loads_mat[:, gid] + float(S_add)
        loads_mat[:, gid] = new_col
        mask = new_col > max_loads
        if np.any(mask): max_loads[mask] = new_col[mask]

        # If still has empty slots, push back to heap with new score
        if len(sim.gpu_active[gid]) < sim.batch_size:
            cur, fut = float(loads_mat[0, gid]), float(loads_mat[-1, gid])
            heapq.heappush(heap_, ((1.0 - lam) * cur + lam * fut, gid))

        round_cnt += 1
