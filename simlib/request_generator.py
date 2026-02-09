import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

def generate_requests(
    N: int,
    mean_s: float = 3200,
    std_s: float = 2600,
    mean_o: float = 1200,
    std_o: float = 700,
    rho: float = -0.1,
    delta_lower: float = 0.0,
    delta_upper: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    max_retry: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 N 个请求对 (s, o, o_est)，满足 s, o > 0，且具有指定相关性 rho。
    预先分解协方差矩阵，使用截尾采样（拒绝负数），确保稳定。
    对输出 o 添加非对称乘性噪声：o_est = round(o * U[1-delta_lower, 1+delta_upper]).
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- 构造均值/协方差，并做基本有限性检查 ---
    cov = rho * std_s * std_o
    sigma = np.array([[std_s**2, cov],
                      [cov,       std_o**2]], dtype=np.float64)
    mu = np.array([mean_s, mean_o], dtype=np.float64)
    if not (np.isfinite(sigma).all() and np.isfinite(mu).all()):
        raise ValueError("mean/std/rho 导致 sigma 或 mu 非有限，请检查入参。")

    # --- Cholesky ---
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        eps = 1e-6 * np.trace(sigma)
        sigma += eps * np.eye(2)
        L = np.linalg.cholesky(sigma)

    # 追加有限性断言，避免后续 matmul 产生 NaN/inf 告警
    if not np.isfinite(L).all():
        raise ValueError("Cholesky 分解结果存在非有限值，请调整 std 或 rho。")

    s_list = np.empty(N, dtype=np.int64)
    o_list = np.empty(N, dtype=np.int64)
    filled = 0
    retry = 0

    while filled < N:
        batch_size = max(1, int((N - filled) * 1.5))

        Z = rng.standard_normal((batch_size, 2))
        Z = np.ascontiguousarray(Z)  # 保留
        samples = np.dot(Z, L.T) + mu
        samples = samples[(samples[:, 0] > 0) & (samples[:, 1] > 0)]

        take = min(samples.shape[0], N - filled)
        s_list[filled:filled+take] = np.round(samples[:take, 0]).astype(np.int64)
        o_list[filled:filled+take] = np.round(samples[:take, 1]).astype(np.int64)
        filled += take

        retry += 1
        if retry > max_retry:
            raise RuntimeError(f"Sampling failed after {max_retry} retries. Try reducing std or rho.")

    if delta_lower < 0 or delta_upper < 0:
        raise ValueError("delta_lower 和 delta_upper 需要为非负数。")

    # 乘性噪声估计
    u = rng.uniform(1.0 - delta_lower, 1.0 + delta_upper, size=N)
    o_est_list = np.maximum(1, np.round(o_list * u)).astype(np.int64)

    return s_list, o_list, o_est_list


def load_sharegpt_dataset(
    dataset_path: str,
    N: int,
    delta_lower: float = 0.0,
    delta_upper: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    tokenizer_name: str = "cl100k_base",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从ShareGPT数据集直接采样N个真实的prefill和decode长度（使用reservoir sampling，不全部加载）。
    
    参数:
        dataset_path: ShareGPT数据集路径（JSON文件或包含JSON文件的目录）
        N: 需要采样的请求数量
        delta_lower: o_est的乘性噪声下界
        delta_upper: o_est的乘性噪声上界
        rng: 随机数生成器
        tokenizer_name: tiktoken编码器名称，默认"cl100k_base"（GPT-4）
    
    返回:
        s_arr, o_arr, o_est_arr: prefill长度、decode长度、估计的decode长度
    """
    if not TIKTOKEN_AVAILABLE:
        raise ImportError("tiktoken is required for loading ShareGPT dataset. Install it with: pip install tiktoken")
    
    if rng is None:
        rng = np.random.default_rng()
    
    # 初始化tokenizer
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer '{tokenizer_name}': {e}")
    
    # 确定要读取的文件
    dataset_path = Path(dataset_path)
    json_files = []
    
    if dataset_path.is_file():
        json_files = [dataset_path]
    elif dataset_path.is_dir():
        json_files = list(dataset_path.glob("*.json"))
    else:
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    if not json_files:
        raise ValueError(f"No JSON files found in {dataset_path}")
    
    # 使用reservoir sampling直接采样N个样本
    reservoir_s = []
    reservoir_o = []
    count = 0
    
    # 遍历所有JSON文件
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                conversations = item.get('conversations', [])
                if not conversations:
                    continue
                
                # 遍历对话对：human -> gpt
                for i in range(0, len(conversations) - 1, 2):
                    if i + 1 >= len(conversations):
                        break
                    
                    human_msg = conversations[i]
                    gpt_msg = conversations[i + 1]
                    
                    if human_msg.get('from') == 'human' and gpt_msg.get('from') == 'gpt':
                        prompt_text = human_msg.get('value', '')
                        response_text = gpt_msg.get('value', '')
                        
                        if prompt_text and response_text:
                            try:
                                prompt_tokens = len(enc.encode(prompt_text))
                                response_tokens = len(enc.encode(response_text))
                                
                                if prompt_tokens > 0 and response_tokens > 0:
                                    count += 1
                                    
                                    # Reservoir sampling算法
                                    if len(reservoir_s) < N:
                                        # 如果reservoir还没满，直接添加
                                        reservoir_s.append(prompt_tokens)
                                        reservoir_o.append(response_tokens)
                                    else:
                                        # 随机决定是否替换
                                        j = rng.integers(0, count)
                                        if j < N:
                                            reservoir_s[j] = prompt_tokens
                                            reservoir_o[j] = response_tokens
                            except Exception:
                                # 跳过编码失败的样本
                                continue
    
    if len(reservoir_s) < N:
        # 如果有效样本数少于N，进行有放回采样
        if len(reservoir_s) == 0:
            raise ValueError("No valid conversation pairs found in the dataset")
        
        indices = rng.choice(len(reservoir_s), size=N, replace=True)
        s_arr = np.array([reservoir_s[i] for i in indices], dtype=np.int64)
        o_arr = np.array([reservoir_o[i] for i in indices], dtype=np.int64)
    else:
        s_arr = np.array(reservoir_s, dtype=np.int64)
        o_arr = np.array(reservoir_o, dtype=np.int64)
    
    # 生成o_est（带噪声的估计值）
    if delta_lower < 0 or delta_upper < 0:
        raise ValueError("delta_lower 和 delta_upper 需要为非负数。")
    
    u = rng.uniform(1.0 - delta_lower, 1.0 + delta_upper, size=N)
    o_est_arr = np.maximum(1, np.round(o_arr * u)).astype(np.int64)
    
    return s_arr, o_arr, o_est_arr


def load_arxiv_dataset(
    dataset_path: str,
    N: int,
    delta_lower: float = 0.0,
    delta_upper: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从arxiv数据集JSON文件加载真实的prefill和decode长度。
    文件格式：{"data": [{"input_length": int, "output_length": int}, ...]}
    
    参数:
        dataset_path: arxiv数据集JSON文件路径
        N: 需要采样的请求数量
        delta_lower: o_est的乘性噪声下界
        delta_upper: o_est的乘性噪声上界
        rng: 随机数生成器
    
    返回:
        s_arr, o_arr, o_est_arr: prefill长度、decode长度、估计的decode长度
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 加载JSON文件
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取数据数组
    if isinstance(data, dict) and 'data' in data:
        data_list = data['data']
    elif isinstance(data, list):
        data_list = data
    else:
        raise ValueError(f"Unexpected data format in {dataset_path}")
    
    # 提取input_length和output_length
    s_list = []
    o_list = []
    
    for item in data_list:
        if 'input_length' in item and 'output_length' in item:
            s = int(item['input_length'])
            o = int(item['output_length'])
            if s > 0 and o > 0:
                s_list.append(s)
                o_list.append(o)
    
    if not s_list:
        raise ValueError("No valid data found in the dataset")
    
    s_arr = np.array(s_list, dtype=np.int64)
    o_arr = np.array(o_list, dtype=np.int64)
    
    # 采样N个样本
    if N > len(s_arr):
        # 有放回采样
        indices = rng.choice(len(s_arr), size=N, replace=True)
    else:
        # 无放回采样
        indices = rng.choice(len(s_arr), size=N, replace=False)
    
    s_arr = s_arr[indices]
    o_arr = o_arr[indices]
    
    # 生成o_est（带噪声的估计值）
    if delta_lower < 0 or delta_upper < 0:
        raise ValueError("delta_lower 和 delta_upper 需要为非负数。")
    
    u = rng.uniform(1.0 - delta_lower, 1.0 + delta_upper, size=N)
    o_est_arr = np.maximum(1, np.round(o_arr * u)).astype(np.int64)
    
    return s_arr, o_arr, o_est_arr


def load_burstgpt_dataset(
    N: int,
    delta_lower: float = 0.0,
    delta_upper: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    dataset_name: str = "lzzmm/burst_gpt",
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从burstGPT数据集加载真实的prefill和decode长度。
    使用HuggingFace datasets库加载，支持reservoir sampling直接采样N个样本。
    
    参数:
        N: 需要采样的请求数量
        delta_lower: o_est的乘性噪声下界
        delta_upper: o_est的乘性噪声上界
        rng: 随机数生成器
        dataset_name: HuggingFace数据集名称，默认"lzzmm/burst_gpt"
        split: 数据集split，默认"train"
    
    返回:
        s_arr, o_arr, o_est_arr: prefill长度、decode长度、估计的decode长度
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library is required for loading burstGPT dataset. Install it with: pip install datasets")
    
    if rng is None:
        rng = np.random.default_rng()
    
    # 加载数据集
    print(f"Loading burstGPT dataset: {dataset_name} (split: {split})...")
    dataset = load_dataset(dataset_name, split=split)
    
    # 使用reservoir sampling直接采样N个样本
    reservoir_s = []
    reservoir_o = []
    count = 0
    
    print(f"Sampling {N} samples from {len(dataset)} total samples...")
    for item in dataset:
        request_tokens = item.get('Request tokens', None)
        response_tokens = item.get('Response tokens', None)
        
        # 处理字符串类型的token数
        if isinstance(request_tokens, str):
            try:
                request_tokens = int(request_tokens)
            except (ValueError, TypeError):
                continue
        if isinstance(response_tokens, str):
            try:
                response_tokens = int(response_tokens)
            except (ValueError, TypeError):
                continue
        
        if request_tokens is None or response_tokens is None:
            continue
        
        s = int(request_tokens)
        o = int(response_tokens)
        
        if s > 0 and o > 0:
            count += 1
            
            # Reservoir sampling算法
            if len(reservoir_s) < N:
                # 如果reservoir还没满，直接添加
                reservoir_s.append(s)
                reservoir_o.append(o)
            else:
                # 随机决定是否替换
                j = rng.integers(0, count)
                if j < N:
                    reservoir_s[j] = s
                    reservoir_o[j] = o
    
    if len(reservoir_s) < N:
        # 如果有效样本数少于N，进行有放回采样
        if len(reservoir_s) == 0:
            raise ValueError("No valid data found in the burstGPT dataset")
        
        print(f"Warning: Only {len(reservoir_s)} valid samples found, using replacement sampling...")
        indices = rng.choice(len(reservoir_s), size=N, replace=True)
        s_arr = np.array([reservoir_s[i] for i in indices], dtype=np.int64)
        o_arr = np.array([reservoir_o[i] for i in indices], dtype=np.int64)
    else:
        s_arr = np.array(reservoir_s, dtype=np.int64)
        o_arr = np.array(reservoir_o, dtype=np.int64)
    
    # 生成o_est（带噪声的估计值）
    if delta_lower < 0 or delta_upper < 0:
        raise ValueError("delta_lower 和 delta_upper 需要为非负数。")
    
    u = rng.uniform(1.0 - delta_lower, 1.0 + delta_upper, size=N)
    o_est_arr = np.maximum(1, np.round(o_arr * u)).astype(np.int64)
    
    print(f"Successfully sampled {len(s_arr)} samples")
    return s_arr, o_arr, o_est_arr
