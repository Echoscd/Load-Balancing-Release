import numpy as np

def policy_round_robin(sim):
    """
    Round Robin strategy: assign requests to GPUs in round-robin order.
    Each time take requests from the waiting queue, assign them to GPU 0, 1, 2, ..., n_worker-1 in sequence, then cycle.
    """
    caps = sim._gpu_capacity_left()
    if not sim.wait_q or caps.sum() == 0:
        return
    
    wait_arr = np.array(list(sim.wait_q), dtype=np.int32)
    total_cap = int(caps.sum())
    num_to_assign = min(len(wait_arr), total_cap)
    
    if num_to_assign == 0:
        return
    
    assigned = []
    gid = 0
    # Use local array to track remaining capacity
    remaining_caps = caps.copy()
    
    for idx in range(num_to_assign):
        # Find next GPU with available capacity
        attempts = 0
        while remaining_caps[gid] <= 0 and attempts < sim.n_worker:
            gid = (gid + 1) % sim.n_worker
            attempts += 1
        
        if remaining_caps[gid] <= 0:
            break
        
        rid = int(wait_arr[idx])
        sim._assign_to_gpu(rid, gid)
        assigned.append(rid)
        remaining_caps[gid] -= 1
        gid = (gid + 1) % sim.n_worker
    
    if assigned:
        sim._remove_from_wait_q(assigned)

