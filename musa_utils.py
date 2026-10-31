"""
MUSA / CUDA 设备适配层。
自动检测 MUSA (摩尔线程) → CUDA (NVIDIA) → CPU。
用法:
    import musa_utils as mu
    mu.setup_distributed()
    mu.set_random_seed(42)
    mu.device
"""
import os
import random
import numpy as np
import torch
import torch.distributed as dist

# ── 自动检测 ──────────────────────────────────────────────
_HAS_MUSA = hasattr(torch, 'musa') and torch.musa.is_available()
_HAS_CUDA = torch.cuda.is_available()

if _HAS_MUSA:
    _backend = 'musa'
    _torch_dev = torch.musa
    DIST_BACKEND = 'mccl'
elif _HAS_CUDA:
    _backend = 'cuda'
    _torch_dev = torch.cuda
    DIST_BACKEND = 'nccl'
else:
    _backend = 'cpu'
    _torch_dev = None
    DIST_BACKEND = 'gloo'

DEVICE_TYPE = _backend

# ── MUSA: 禁用 Flash SDPA（MUDNN 不支持 dropout 下的 Flash Fwd）──
if _HAS_MUSA:
    # 方法 1：后端 API 开关（先检查是否有 musa backend）
    _backends = [torch.backends.cuda]
    if hasattr(torch.backends, 'musa'):
        _backends.insert(0, torch.backends.musa)
    for _api in _backends:
        try:
            _api.enable_flash_sdp(False)
            _api.enable_mem_efficient_sdp(False)
            _api.enable_math_sdp(True)
            print(f"[musa_utils] Flash SDPA disabled via {_api}")
        except Exception:
            pass
    # 方法 2：环境变量兜底
    os.environ.setdefault("SDPA_BACKEND", "math")
    os.environ.setdefault("TORCH_MUSA_ENABLE_FLASH_SDPA", "0")
    # 方法 3：monkey-patch SDPA 强制 math 后端
    try:
        import torch.nn.functional as _F
        _orig_sdpa = _F.scaled_dot_product_attention
        def _math_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None, enable_gqa=False):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                  dropout_p=dropout_p, is_causal=is_causal,
                                  scale=scale, enable_gqa=enable_gqa)
        _F.scaled_dot_product_attention = _math_sdpa
        print("[musa_utils] SDPA monkey-patched to force math backend")
    except Exception as _e:
        print(f"[musa_utils] SDPA monkey-patch failed: {_e}")


def is_available() -> bool:
    return _torch_dev is not None


def current_device() -> int:
    return _torch_dev.current_device() if _torch_dev else 0


def set_device(idx: int):
    if _torch_dev:
        _torch_dev.set_device(idx)


def is_bf16_supported() -> bool:
    """仅 MUSA ≥S5000 和 CUDA ≥Ampere 支持 bf16"""
    if _torch_dev is None:
        return False
    return _torch_dev.is_bf16_supported()


def mem_get_info(dev_idx: int = 0):
    if _torch_dev is None:
        return (0, 0)
    return _torch_dev.mem_get_info(dev_idx)


def empty_cache():
    """清空 GPU 缓存（MUSA/CUDA 通用）"""
    if _torch_dev:
        _torch_dev.empty_cache()


def manual_seed(seed: int):
    if _HAS_MUSA:
        torch.musa.manual_seed(seed)
        torch.musa.manual_seed_all(seed)
    if _HAS_CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ── 高层封装 ──────────────────────────────────────────────

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manual_seed(seed)
    if _HAS_CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        torch.manual_seed(seed + rank)


def setup_distributed():
    """初始化分布式，返回 (distributed, local_rank, device)"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=DIST_BACKEND)
        local_rank = int(os.environ["LOCAL_RANK"])
        set_device(local_rank)
        dev = torch.device(_backend, local_rank)
        distributed = True
        print(f"[Init] DDP on rank {dist.get_rank()}/{dist.get_world_size()} "
              f"(GPU {local_rank}, backend={DIST_BACKEND})")
    else:
        dev = torch.device(_backend) if is_available() else torch.device("cpu")
        distributed = False
        local_rank = 0
        print(f"[Init] Single-GPU on {dev}.")
    return distributed, local_rank, dev


def reserve_gpu_memory(proportion: float = 0.95):
    """预占当前 GPU 剩余显存"""
    if _torch_dev is None:
        return
    idx = current_device()
    free, total = _torch_dev.mem_get_info(idx)
    size = int(free * proportion)
    print(f"剩余显存: {free / 1024**3:.2f} GB, 预占: {size / 1024**3:.2f} GB")
    dummy = torch.empty(size, dtype=torch.uint8, device=f'{_backend}:{idx}')
    del dummy
    print("显存预占成功。")


# ── 设备字符串 ────────────────────────────────────────────
DEFAULT_DEVICE = f"{_backend}:0" if is_available() else "cpu"
