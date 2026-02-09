import numpy as np

def policy_FCFS(sim):
    """FCFS：按进入顺序取请求，依次分配到 GPU。"""
    caps = sim._gpu_capacity_left()
    if not sim.wait_q or caps.sum() == 0: return
    ids = np.fromiter(sim.wait_q, dtype=np.int32)[:caps.sum()]
    assigned = []; i = 0
    for gid, cap in enumerate(caps):
        for rid in ids[i:i+cap]:
            sim._assign_to_gpu(int(rid), gid); assigned.append(int(rid))
        i += cap
        if i >= len(ids): break
    if assigned: sim._remove_from_wait_q(assigned)