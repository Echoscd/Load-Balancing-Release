from __future__ import annotations
import heapq
import numpy as np
from typing import List, Tuple

EPS = 1e-6

def policy_lpt_heap(sim):
    """效率优先 → LPT 排序 → 堆负载均衡。"""
    if not sim.wait_q: return
    caps = sim._gpu_capacity_left(); slots = int(caps.sum())
    if slots == 0: return

    # 动态 λ：当前时间 vs 预计完成
    finish = np.array([
        (sim.s_arr[g] + sim.o_est_arr[g]).sum() if g.size else 0.0
        for g in sim.gpu_active
    ], dtype=np.float64)
    est = (finish.max() * sim.t_length) if finish.size else 1.0
    lam = min(1.0, sim.clock / (est + EPS))

    # 候选：效率 top-k，再按 LPT 排序
    ids = np.fromiter(sim.wait_q, dtype=np.int32)
    k = min(slots, ids.size)
    eff = sim.eff_arr[ids]
    top = ids[np.argpartition(-eff, k-1)[:k]]
    top = top[np.argsort(-sim.s_arr[top])]

    # 堆初始化：GPU 当前+未来负载
    heap: List[Tuple[float,int]] = []
    for gid in range(sim.n_worker):
        if caps[gid] <= 0: continue
        g = sim.gpu_active[gid]
        cur = sim.s_arr[g].sum() if g.size else 0.0
        fut = (sim.s_arr[g] + sim.o_est_arr[g]).sum() if g.size else 0.0
        score = (1-lam)*cur + lam*fut
        heapq.heappush(heap,(score,gid))

    # 分配循环
    assigned = []
    for rid in top:
        while heap and caps[heap[0][1]]==0: heapq.heappop(heap)
        if not heap: break
        _,gid = heapq.heappop(heap)
        if caps[gid]<=0: continue
        sim._assign_to_gpu(int(rid),gid); assigned.append(int(rid)); caps[gid]-=1
        g = sim.gpu_active[gid]
        cur,fut = sim.s_arr[g].sum(),(sim.s_arr[g]+sim.o_est_arr[g]).sum()
        new_score = (1-lam)*cur+lam*fut
        heapq.heappush(heap,(new_score,gid))

    if assigned: sim._remove_from_wait_q(assigned)
