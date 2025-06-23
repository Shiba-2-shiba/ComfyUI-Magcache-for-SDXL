# magcache_core.py

import torch
import numpy as np
import json
import os
import hashlib
import weakref
from comfy import model_management # CORRECTED IMPORT

CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "magcache_data")
os.makedirs(CALIBRATION_DIR, exist_ok=True)

# 1. モデルハッシュ取得関数
def get_model_hash(model_name: str) -> str:
    return hashlib.sha256(model_name.encode('utf-8')).hexdigest()[:16]

# 2. mag_ratios のロード/セーブ関数
def load_mag_ratios(model_hash: str):
    filepath = os.path.join(CALIBRATION_DIR, f"{model_hash}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return np.array(data['mag_ratios'])
        except Exception:
            return None
    return None

def save_mag_ratios(model_hash: str, ratios: list, model_name: str):
    filepath = os.path.join(CALIBRATION_DIR, f"{model_hash}.json")
    data = {
        'model_name': model_name,
        'mag_ratios': ratios
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[MagCache-SDXL] Calibration data saved to {filepath}")

# 3. mag_ratios 補間関数
def interpolate_mag_ratios(ratios, target_steps):
    if ratios is None or len(ratios) == 0:
        return None
    
    source_steps = len(ratios) // 2
    if source_steps == target_steps:
        return ratios

    cond_ratios = ratios[0::2]
    uncond_ratios = ratios[1::2]
    
    source_indices = np.linspace(0, source_steps - 1, source_steps)
    target_indices = np.linspace(0, source_steps - 1, target_steps)
    
    interp_cond = np.interp(target_indices, source_indices, cond_ratios)
    interp_uncond = np.interp(target_indices, source_indices, uncond_ratios)

    result = np.empty((target_steps * 2,), dtype=ratios.dtype)
    result[0::2] = interp_cond
    result[1::2] = interp_uncond
    
    print(f"[MagCache-SDXL] Ratios interpolated from {source_steps} to {target_steps} steps.")
    return result

# 4. 状態管理クラス
class MagCacheState:
    def __init__(self):
        self.state = {
            "cond": self._create_guidance_state(),
            "uncond": self._create_guidance_state()
        }
        self.cache_device = model_management.get_torch_device()

    def _create_guidance_state(self):
        return {'residual_cache': None, 'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0}
    
    def reset(self):
        self.state = {
            "cond": self._create_guidance_state(),
            "uncond": self._create_guidance_state()
        }
    
    def get_state(self, guidance_type: str):
        return self.state[guidance_type]

    def store_residual(self, residual_tensor: torch.Tensor, guidance_type: str):
        state = self.get_state(guidance_type)
        state['residual_cache'] = residual_tensor.to(device=self.cache_device, copy=True)