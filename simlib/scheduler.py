from __future__ import annotations
import numpy as np
from collections import deque
from typing import Callable, List, Dict, Optional, Iterable


class SchedulerSim:
    """
    在线调度模拟器：
    - 请求预生成，逐步揭示 (reveal)
    - 始终保持至少 reveal_target 个候选请求
    - policy_fn 负责将请求分配到 GPU
    """

    def __init__(
        self,
        n_worker: int,
        batch_size: int,
        C: float,
        t_length: float,
        s_list: np.ndarray,
        o_list: np.ndarray,
        o_est_list: np.ndarray,
        policy_fn: Callable[["SchedulerSim"], None],
        eff_name: str = "ratio",
        record_history: bool = True,
        reveal_target: Optional[int] = None,
        discover_fn: Optional[Callable[["SchedulerSim", int], List[int]]] = None,
        n_params: Optional[int] = None,  # 模型参数数量，用于计算MFU
        flops_peak: Optional[float] = None,  # GPU理论峰值FLOPS (TFLOPs)，用于计算MFU
    ):
        # 基本配置
        self.n_worker = int(n_worker)      # GPU 数
        self.batch_size = int(batch_size)  # 每块 GPU 最大并行并发数
        self.C = float(C)                  # 固定开销
        self.t_length = float(t_length)    # 单 token 生成时延
        self.policy_fn = policy_fn
        self.record_hist = record_history
        
        # Power计算参数（根据论文D.1节）
        self.n_params = int(n_params) if n_params is not None else 70_000_000_000  # 默认70B参数
        self.flops_peak = float(flops_peak) if flops_peak is not None else 312.0  # 默认A100: 312 TFLOPs
        # Power模型参数
        self.P_idle = 100.0  # W
        self.P_max = 400.0    # W
        self.mfu_sat = 0.45   # MFU饱和点
        self.gamma = 0.7      # 次线性缩放指数

        # 请求参数
        from .efficiency import EFF_FUNCS
        eff_fn = EFF_FUNCS[eff_name]
        self.N = len(s_list)
        self.s_arr = s_list.astype(np.int32, copy=True)          # prompt 长度
        self.o_arr = o_list.astype(np.int32, copy=True)          # output 长度
        self.o_est_arr = o_est_list.astype(np.int32, copy=True)  # output 估计值（供策略使用）
        self.gen_arr = np.zeros(self.N, dtype=np.int32)          # 已生成 token 数
        self.eff_arr = eff_fn(self.s_arr, self.o_arr).astype(np.float32)

        # 请求状态
        self.gpu_of_req = -np.ones(self.N, dtype=np.int32)       # 请求 -> GPU
        self.start_time = np.full(self.N, np.nan, dtype=np.float64)
        self.finish_time = np.full(self.N, np.nan, dtype=np.float64)
        self._unfinished_mask = np.ones(self.N, dtype=bool)

        # 在线队列
        self.undiscovered_q: deque[int] = deque(range(self.N))   # 未揭示的请求
        self.wait_q: deque[int] = deque()                        # 候选请求
        self.reveal_target = reveal_target if reveal_target is not None else (self.n_worker * 4)
        self.discover_fn = discover_fn if discover_fn is not None else self.discover_fn_default

        # GPU 状态
        self.gpu_active: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(self.n_worker)]
        self.sum_lengths = np.zeros(self.n_worker, dtype=np.int64)
        self.active_total = 0

        # 时间步
        self.clock = 0.0
        self.step_idx = 0
        self.last_assign_step = -1

        # 历史记录（仅在 record_hist=True 时填充）
        self.hist_sum_lengths: List[np.ndarray] = []
        self.hist_steps: List[int] = []

        self._reveal_next()  # 初始揭示

    # =============== 类内默认发现函数 =============== #
    @staticmethod
    def discover_fn_default(sim: "SchedulerSim", needed: int) -> List[int]:
        """默认发现：从 undiscovered_q 中取出 needed 个请求。"""
        out = []
        for _ in range(min(needed, len(sim.undiscovered_q))):
            out.append(sim.undiscovered_q.popleft())
        return out

    # ================== 内部辅助函数 ================== #
    def _reveal_next(self):
        """保证 wait_q 至少有 reveal_target 个请求。"""
        if not self.undiscovered_q:
            return
        need = self.reveal_target - len(self.wait_q)
        if need <= 0:
            return
        new_ids = self.discover_fn(self, need)
        self.wait_q.extend(new_ids)

    def _gpu_capacity_left(self) -> np.ndarray:
        """返回每个 GPU 剩余可分配的 slot 数。"""
        return self.batch_size - np.array([len(a) for a in self.gpu_active], dtype=np.int32)

    def _remove_from_wait_q(self, ids: Iterable[int]):
        """批量删除 wait_q 中的已分配请求。"""
        ids_set = set(int(x) for x in ids)
        if not ids_set:
            return
        self.wait_q = type(self.wait_q)([x for x in self.wait_q if x not in ids_set])

    def _assign_to_gpu(self, req_id: int, gpu_id: int):
        """将请求分配给指定 GPU。"""
        if len(self.gpu_active[gpu_id]) >= self.batch_size:
            raise AssertionError(f"GPU {gpu_id} exceeds batch_size={self.batch_size}")
        self.gpu_active[gpu_id] = np.append(self.gpu_active[gpu_id], req_id)
        self.sum_lengths[gpu_id] += self.s_arr[req_id] + self.gen_arr[req_id]
        self.gpu_of_req[req_id] = gpu_id
        self.active_total += 1
        if np.isnan(self.start_time[req_id]):
            self.start_time[req_id] = self.clock
        self.last_assign_step = self.step_idx

    def _remove_finished_from_gpu(self, gpu_id: int, finished_ids: np.ndarray):
        """从 GPU 中移除已完成的请求。"""
        if finished_ids.size == 0:
            return
        active_ids = self.gpu_active[gpu_id]
        remain_ids = active_ids[~np.isin(active_ids, finished_ids)]
        self.gpu_active[gpu_id] = remain_ids
        self.sum_lengths[gpu_id] = int((self.s_arr[remain_ids] + self.gen_arr[remain_ids]).sum())
        removed = int(active_ids.size - remain_ids.size)
        if removed:
            self.active_total -= removed

    # ================== 主循环 ================== #
    def run(self, verbose: bool = False) -> Dict[str, float]:
        """
        执行调度模拟并返回指标：
        makespan、avg_latency、avg_latency_per_token、throughput_tokens_per_sec、QPM、avg_gpu_imbalance、avg_gpu_utilization
        """
        self._reveal_next()
        self.policy_fn(self)

        finished_cnt = 0
        step_active: List[int] = []
        step_finished: List[int] = []
        step_time: List[float] = []
        step_imb: List[float] = []
        step_util: List[float] = []
        step_idle_rate: List[float] = []
        step_power: List[float] = []  # 每个step的总energy (J = W·s)
        step_avg_power: List[float] = []  # 每个step的平均power (W)
        step_clock_time: List[float] = []  # 每个step的时间戳

        policy_fn = self.policy_fn
        sum_lengths = self.sum_lengths
        gen_arr = self.gen_arr
        s_arr = self.s_arr
        o_arr = self.o_arr
        start_time = self.start_time
        finish_time = self.finish_time
        unfinished_mask = self._unfinished_mask
        total_capacity = float(self.n_worker * self.batch_size)
        gpu_active = self.gpu_active

        while finished_cnt < self.N:
            active_now = self.active_total
            if active_now == 0:
                # 尝试填充 GPU
                if self.wait_q or self.undiscovered_q:
                    self._reveal_next()
                    policy_fn(self)
                    active_now = self.active_total
                if active_now == 0:
                    break  # 完全空闲

            # 时间推进：固定开销 + 最大负载 * 单 token 时间
            max_sum_before = sum_lengths.max()
            step_dur = self.C + self.t_length * max_sum_before
            #print(max_sum_before, step_dur)
            #print(self.C, self.t_length * max_sum_before)
            self.clock += step_dur
            self.step_idx += 1

            # 每个活跃请求生成一个 token
            for g in range(self.n_worker):
                ids = gpu_active[g]
                if ids.size:
                    gen_arr[ids] += 1
                    sum_lengths[g] += ids.size

            if self.record_hist:
                self.hist_sum_lengths.append(sum_lengths.copy())
                self.hist_steps.append(self.step_idx)

            # 检查完成请求
            newly_finished = np.where(unfinished_mask & (gen_arr >= o_arr))[0]
            if newly_finished.size:
                finish_time[newly_finished] = self.clock
                unfinished_mask[newly_finished] = False
                finished_cnt += newly_finished.size
                for g in range(self.n_worker):
                    fin_g = newly_finished[self.gpu_of_req[newly_finished] == g]
                    if fin_g.size:
                        self._remove_finished_from_gpu(g, fin_g)

            # 记录统计
            current_max = sum_lengths.max()
            imb = self.n_worker * current_max - sum_lengths.sum()
            util = active_now / total_capacity
            
            # 计算GPU idle率：每个step中 Average(1 - 该GPU处理的token数/最大GPU_token)
            # 每个GPU处理的token数 = 该GPU上所有活跃请求的token总和 (s + gen)
            gpu_tokens = np.zeros(self.n_worker, dtype=np.int64)
            for g in range(self.n_worker):
                ids = gpu_active[g]
                if ids.size > 0:
                    gpu_tokens[g] = int((s_arr[ids] + gen_arr[ids]).sum())
            max_gpu_tokens = gpu_tokens.max() if gpu_tokens.max() > 0 else 1
            # 对于每个GPU：idle_rate = 1 - (该GPU处理的token数 / 最大GPU_token)
            gpu_idle_rates = 1.0 - (gpu_tokens.astype(np.float64) / max_gpu_tokens)
            # 对所有GPU求平均
            step_idle = float(gpu_idle_rates.mean())
            
            # 计算Power和Energy：根据公式 P(mfu) = P_idle + (P_max - P_idle) * (mfu / mfu_sat)^γ
            # 使用idle率替代 mfu/mfu_sat: utilization_ratio = 1 - idle_rate
            # Energy = Σ_g [P(mfu_g) * Δt]  (公式D47)
            step_energy = 0.0
            gpu_powers = np.zeros(self.n_worker, dtype=np.float64)
            if step_dur > 0:
                for g in range(self.n_worker):
                    # 使用idle率计算utilization ratio: utilization = 1 - idle_rate
                    utilization_ratio = 1.0 - gpu_idle_rates[g]
                    # 限制utilization ratio在合理范围内 [0, 1]
                    utilization_ratio = max(0.0, min(utilization_ratio, 1.0))
                    # 计算Power: P = P_idle + (P_max - P_idle) * (utilization_ratio)^γ
                    # 这里用utilization_ratio替代了原来的 mfu/mfu_sat
                    if utilization_ratio > 0:
                        power_g = self.P_idle + (self.P_max - self.P_idle) * (utilization_ratio ** self.gamma)
                    else:
                        power_g = self.P_idle  # idle时使用P_idle
                    gpu_powers[g] = power_g
                    # 累加该GPU的energy: P * Δt
                    step_energy += power_g * step_dur
            else:
                # step_dur为0时，所有GPU都是idle
                gpu_powers.fill(self.P_idle)
                step_energy = 0.0
            # 计算step的平均power
            step_avg_power_val = float(gpu_powers.mean())
            
            step_active.append(active_now)
            step_finished.append(newly_finished.size)
            step_time.append(step_dur)
            step_imb.append(imb)
            step_util.append(util)
            step_idle_rate.append(step_idle)
            step_power.append(step_energy)  # 存储step的energy
            step_avg_power.append(step_avg_power_val)  # 存储step的平均power
            step_clock_time.append(self.clock)  # 存储step的时间戳

            # 继续填充
            if self.wait_q or self.undiscovered_q:
                self._reveal_next()
                policy_fn(self)

            if verbose and self.step_idx % 100 == 0:
                print(f"step {self.step_idx:>4d}, t={self.clock:9.3f}, finished={finished_cnt}/{self.N}")

        if np.isnan(finish_time).any():
            raise RuntimeError("Some requests never finished")

        # 汇总指标
        makespan = float(finish_time.max())
        avg_latency = float(finish_time.mean())
        valid = o_arr > 0
        lat_per_tok = float(((finish_time[valid] - start_time[valid]) / o_arr[valid]).mean())

        cut = max(0, min(int(self.last_assign_step), len(step_time)))
        if cut > 0:
            time_sum = float(np.sum(step_time[:cut]))
            tok_sum = int(np.sum(step_active[:cut]))
            finish_sum = int(np.sum(step_finished[:cut]))
            imb_sum = float(np.sum(step_imb[:cut]))
            util_sum = float(np.sum(step_util[:cut]))
            idle_rate_sum = float(np.sum(step_idle_rate[:cut]))
            energy_sum = float(np.sum(step_power[:cut]))  # step_power存储的是energy
            avg_power_sum = float(np.sum(step_avg_power[:cut]))
            steps_cnt = cut
        else:
            time_sum = tok_sum = finish_sum = imb_sum = util_sum = idle_rate_sum = energy_sum = avg_power_sum = 0.0
            steps_cnt = 0

        throughput_metric = (tok_sum / time_sum) if time_sum else 0.0
        qpm_metric = (60.0 * finish_sum / time_sum) if time_sum else 0.0
        avg_imbalance = (imb_sum / steps_cnt) if steps_cnt else 0.0
        avg_gpu_util = (util_sum / steps_cnt) if steps_cnt else 0.0
        avg_gpu_idle_rate = (idle_rate_sum / steps_cnt) if steps_cnt else 0.0
        total_energy = energy_sum  # 总energy (J)
        avg_power = (avg_power_sum / steps_cnt) if steps_cnt else self.P_idle  # 平均power (W)

        return dict(
            makespan=makespan,
            avg_latency=avg_latency,
            avg_latency_per_token=lat_per_tok,
            throughput_tokens_per_sec=throughput_metric,
            QPM=qpm_metric,
            avg_gpu_imbalance=avg_imbalance,
            avg_gpu_utilization=avg_gpu_util,
            avg_gpu_idle_rate=avg_gpu_idle_rate,
            total_energy=total_energy,
            avg_power=avg_power,
            step_clock_times=step_clock_time[:cut] if cut > 0 else [],  # 返回时间戳列表
            step_powers=step_avg_power[:cut] if cut > 0 else [],  # 返回瞬时功率列表
        )
