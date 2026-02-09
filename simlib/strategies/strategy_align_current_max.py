import numpy as np

EPS = 1e-6
MAX_CANDIDATES = 256 * 4         # 每轮最多参与的候选请求

# ========= 只看“当前步骤 max”的对齐策略 =========
def _pick_rid_by_gap(sim, wait_ids, gap_int: int) -> tuple[int, int]:
    """
    在 wait_ids 中挑一个请求：
      - 若存在 s <= gap_int，则取“最大的 s”
      - 否则取“最小的 s”
    返回 (rid, s)。若 wait_ids 为空，返回 (-1, 0)
    """
    if not wait_ids:
        return -1, 0

    best_fit_idx = -1
    best_fit_s = -1
    min_idx = -1
    min_s = None

    for i, rid in enumerate(wait_ids):
        s = int(sim.s_arr[int(rid)])

        # 记录最小 s（兜底用）
        if (min_s is None) or (s < min_s):
            min_s = s
            min_idx = i

        # 记录不超过 gap 的最大 s（首选）
        if gap_int >= 1 and s <= gap_int and s > best_fit_s:
            best_fit_s = s
            best_fit_idx = i

    if best_fit_idx >= 0:
        rid = int(wait_ids[best_fit_idx])
        return rid, int(best_fit_s)

    # 无法做到“严格小于 M0”，退而求其次取最小 s
    rid = int(wait_ids[min_idx])
    return rid, int(min_s if min_s is not None else 0)


def policy_align_current_max(sim) -> None:
    """
    固定当前步骤的全局最大负载 M0，只看当前，不看未来。
    循环为有空位的 GPU 分配：
      - 目标：令该 GPU 的新负载 < M0 且尽量接近 M0
      - 若做不到，则给它分配 wait 队列里最小的 s
    每次只分配 1 个请求，重复直到无空位或无等待。
    """
    if not sim.wait_q:
        return

    # 当前各 GPU 负载与“本步骤”固定的全局最大负载
    loads = np.asarray(sim.sum_lengths, dtype=np.float64).copy()
    M0 = float(loads.max())

    # 建一个按“当前负载”排序的最小堆（只放有空位的 GPU）
    import heapq
    cap_left = sim._gpu_capacity_left()
    heap = []
    for gid in range(int(sim.n_worker)):
        if int(cap_left[gid]) > 0:
            heapq.heappush(heap, (float(loads[gid]), int(gid)))

    # 主循环：每次从当前最空的 GPU 开始补
    while heap and sim.wait_q:
        cur_load, gid = heapq.heappop(heap)

        # 可能在堆期间槽位变化，重新确认
        cap_left = sim._gpu_capacity_left()
        if int(cap_left[gid]) <= 0:
            continue

        # 候选集合（出于速度考虑只看前 MAX_CANDIDATES 个）
        cand_num = min(MAX_CANDIDATES, len(sim.wait_q))
        wait_ids = list(sim.wait_q)[:cand_num]
        if not wait_ids:
            break

        # 目标：new_load = cur_load + s < M0 且尽量接近
        gap_float = M0 - float(cur_load) - EPS
        gap_int = int(np.floor(gap_float))  # 只允许严格小于 M0
        rid, s_chosen = _pick_rid_by_gap(sim, wait_ids, gap_int)
        if rid < 0:
            break  # 理论上不会发生

        # 执行分配
        sim._assign_to_gpu(int(rid), int(gid))
        sim._remove_from_wait_q([int(rid)])

        # 更新该 GPU 的“当前负载”，注意 M0 在本策略内保持固定不变
        new_load = float(cur_load + s_chosen)
        loads[gid] = new_load

        # 若该 GPU 仍有空位，则按新负载回堆，继续“向 M0 靠拢”
        if int(sim._gpu_capacity_left()[gid]) > 0:
            heapq.heappush(heap, (new_load, gid))
