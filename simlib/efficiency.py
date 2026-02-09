
import numpy as np

EPS = 1e-5

def eff_ratio(s, o):
    """标准比例：偏好快结束的请求"""
    return o / (s + o + EPS)

def eff_inverse_volume(s, o):
    """比例除以体量开方：减少大任务聚集"""
    return (o / (s + o + EPS)) / np.sqrt(s + o + EPS)

def eff_symmetry_log_ratio(s, o):
    """鼓励 log(s) 和 log(o) 相近，输入输出平衡"""
    log_s = np.log1p(s)
    log_o = np.log1p(o)
    return 1 - np.abs(log_s - log_o) / (log_s + log_o + EPS)

def eff_per_gpu_token(s, o):
    """每单位 token 负载的有效率，模拟GPU负载影响"""
    return o / (s + o + EPS) / (0.8 * s + 0.2 * o + EPS)

def eff_centered_volume(s, o, center=6000):
    """偏好总 token 长度为中等值（如6000）的请求"""
    total = s + o
    scale = 1 / (1 + np.exp((total - center) / 1000))
    return (o / (s + EPS)) * scale

def eff_kernelized(s, o, mu=5000, sigma=1500):
    """高斯核窗口函数，偏好多样性"""
    total = s + o
    return np.exp(-((total - mu) ** 2) / (2 * sigma ** 2))

def eff_balanced_sigmoid(s, o):
    """比例接近1最优的 sigmoid 模型"""
    ratio = o / (s + EPS)
    return 1 / (1 + np.exp(-10 * (ratio - 1)))

def eff_equalized_weight(s, o):
    """压缩任务体量影响，按比例权重调整"""
    return o / np.sqrt(s + o + EPS)

def eff_doubly_scaled(s, o):
    """比例 × exp(-体量)：避免过长请求集中"""
    total = s + o
    return (o / (s + EPS)) * np.exp(-total / 5000)

def eff_token_per_unit_cost(s, o, a=0.8, b=0.2):
    """按单位资源消耗分配：避免偏向输入型请求"""
    return o / (a * s + b * o + EPS)

def eff_ratio_gaussian(s, o, mu=6000, sigma=1500):
    """比例 × 高斯体量权重，鼓励中等请求"""
    total = s + o
    ratio = o / (s + EPS)
    weight = np.exp(-((total - mu) ** 2) / (2 * sigma ** 2))
    return ratio * weight

EFF_FUNCS = {
    "ratio": eff_ratio,
    "inverse_volume": eff_inverse_volume,
    "symmetry_log_ratio": eff_symmetry_log_ratio,
    "per_gpu_token": eff_per_gpu_token,
    "centered_volume": eff_centered_volume,
    "kernelized": eff_kernelized,
    "balanced_sigmoid": eff_balanced_sigmoid,
    "equalized_weight": eff_equalized_weight,
    "doubly_scaled": eff_doubly_scaled,
    "token_per_unit_cost": eff_token_per_unit_cost,
    "ratio_gaussian": eff_ratio_gaussian,
}
