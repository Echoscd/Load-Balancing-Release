import numpy as np

def policy_join_shortest_queue(sim):
    """
    Join Shortest Queue (JSQ) strategy:
    For each request in the waiting queue, assign it to the GPU with the minimum current token count.
    If multiple GPUs have the same minimum token count, select the first one.
    """
    caps = sim._gpu_capacity_left()
    if not sim.wait_q or caps.sum() == 0:
        return
    
    assigned = []
    wait_list = list(sim.wait_q)
    
    for rid in wait_list:
        # Find GPU with minimum token count (only consider GPUs with capacity)
        valid_gpus = np.where(caps > 0)[0]
        if len(valid_gpus) == 0:
            break
        
        # Find GPU with minimum token count among valid GPUs
        # Use current total token load for each GPU
        valid_token_loads = sim.sum_lengths[valid_gpus]
        min_load_idx = np.argmin(valid_token_loads)
        gid = valid_gpus[min_load_idx]
        
        # Assign request to this GPU
        sim._assign_to_gpu(int(rid), gid)
        assigned.append(int(rid))
        
        # Update capacity (token count will be automatically updated in _assign_to_gpu)
        caps[gid] -= 1
    
    if assigned:
        sim._remove_from_wait_q(assigned)

