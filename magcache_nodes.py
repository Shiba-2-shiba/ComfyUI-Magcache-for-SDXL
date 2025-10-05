# magcache_nodes.py (リファクタリング版)
#
# 目的：
# - UX改善のため、キャッシュの適用状態をノード上に表示する。
# - 初心者向けの簡易設定ノード「MagCacheEasyApply」を追加する。
# - 共通ロジックをまとめてコードの可読性と保守性を向上させる。

import torch
import numpy as np
import json
import atexit
import weakref

# --- コア機能とヘルパーをインポート ---
from .magcache_core import (
    get_model_hash, load_mag_ratios, save_mag_ratios,
    interpolate_mag_ratios, MagCacheState
)

# --- ヘルパー関数 (変更なし) ---
def _get_model_name(model):
    model_name = getattr(model, "magcache_source_ckpt_name", None)
    if not model_name:
        print("[MagCache] Warning: Could not find patched model name.")
    return model_name

def _get_cache_key(kwargs):
    c_kwargs = kwargs.get('c', {})
    cond_or_uncond_list = c_kwargs.get('transformer_options', {}).get('cond_or_uncond', [1])
    guidance_type = "cond" if cond_or_uncond_list[0] == 0 else "uncond"
    ref_suffix = "ref" if 'ref_latents' in c_kwargs else "noref"
    return f"{guidance_type}_{ref_suffix}"


# --- キャリブレーション用ロジック (変更なし) ---
def _perform_calibration_logic(diffusion_model, computed_output, **kwargs):
    try:
        calibration_data = getattr(diffusion_model, 'magcache_calibration_data', None)
        if calibration_data is None: return

        key_prefix = _get_cache_key(kwargs)
        current_eps = computed_output.mean(dim=[1, 2, 3], keepdim=True).detach()
        last_eps_key = f"last_{key_prefix}_eps"
        data_list_key = key_prefix
        
        last_eps = calibration_data.get(last_eps_key)
        if last_eps is not None and last_eps.shape == current_eps.shape:
            ratio = torch.linalg.norm(current_eps) / (torch.linalg.norm(last_eps) + 1e-9)
            calibration_data[data_list_key].append(ratio.item())
        
        calibration_data[last_eps_key] = current_eps
    except Exception as e:
        import traceback
        print(f"[MagCache] Calibration logic error: {e}\n{traceback.format_exc()}")

# --- 終了時にデータを保存するためのハンドラ (変更なし) ---
_atexit_data_store = {}
def _final_save_on_exit():
    model_obj = _atexit_data_store.get('model_ref', lambda: None)()
    if model_obj and getattr(model_obj, 'magcache_is_calibrating', False) and getattr(model_obj, 'magcache_last_run_id', None) is not None:
        print("[MagCache] Process exiting. Performing final save for calibration data.")
        prev_data = getattr(model_obj, 'magcache_calibration_data', {})
        prev_model_name = getattr(model_obj, 'magcache_model_name', "unknown")
        cond_ratios = prev_data.get('cond_noref', []) + prev_data.get('cond_ref', [])
        uncond_ratios = prev_data.get('uncond_noref', []) + prev_data.get('uncond_ref', [])
        min_len = min(len(cond_ratios), len(uncond_ratios))
        if min_len > 0:
            interleaved = np.empty((min_len * 2,), dtype=np.float32)
            interleaved[0::2] = cond_ratios[:min_len]
            interleaved[1::2] = uncond_ratios[:min_len]
            model_hash = get_model_hash(prev_model_name)
            save_mag_ratios(model_hash, interleaved.tolist(), prev_model_name)
        model_obj.magcache_last_run_id = None

if not hasattr(atexit, '_magcache_registered'):
    atexit.register(_final_save_on_exit)
    atexit._magcache_registered = True

# --- パッチ用のフォワード関数 (Apply用・変更なし) ---
def universal_magcache_forward(self, *args, **kwargs):
    params = getattr(self, 'magcache_params', None)
    state_manager = getattr(self, 'magcache_state', None)
    if not params or not state_manager:
        return self.magcache_original_forward(*args, **kwargs)

    current_step = getattr(self, 'magcache_current_step', 0)
    cache_key = _get_cache_key(kwargs)
    state = state_manager.get_state(cache_key)
    
    skip_this_step = False
    if current_step >= params['start_step_abs'] and current_step < params['end_step_abs']:
        ratio_offset = 0 if 'cond' in cache_key else 1
        ratio_index = current_step * 2 + ratio_offset
        if ratio_index < len(params['mag_ratios']) and state.get('residual_cache') is not None:
            ratio = params['mag_ratios'][ratio_index]
            state['accumulated_ratio'] *= ratio
            state['accumulated_steps'] += 1
            state['accumulated_err'] += abs(1.0 - state['accumulated_ratio'])
            if state['accumulated_err'] < params['delta_threshold'] and state['accumulated_steps'] <= params['K_skips']:
                skip_this_step = True
    
    if skip_this_step:
        return state['residual_cache'].to(args[0].device, args[0].dtype)
    else:
        state.update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
        computed_output = self.magcache_original_forward(*args, **kwargs)
        state_manager.store_residual(computed_output.detach(), cache_key)
        return computed_output

# --- [新規] MagCacheApplyとEasyApplyで共有するコアロジック ---
def _apply_magcache_logic(model, enabled, magcache_thresh, magcache_k, retention_ratio, exclude_last_step):
    diffusion_model = model.get_model_object("diffusion_model")
    
    # 既存のパッチを解除してクリーンな状態に戻す
    if hasattr(diffusion_model, 'magcache_original_forward'):
        diffusion_model.forward = diffusion_model.magcache_original_forward
        delattr(diffusion_model, 'magcache_original_forward')
    new_model = model.clone()
    new_model.set_model_unet_function_wrapper(None)
    
    if not enabled:
        print("[MagCache Apply] Disabled.")
        return {"ui": {"text": ["Cache Disabled"]}, "result": (new_model,)}

    model_name = _get_model_name(model)
    if not model_name:
        return {"ui": {"text": ["Cache Inactive: Model name not found"]}, "result": (new_model,)}
    
    model_hash = get_model_hash(model_name)
    mag_ratios = load_mag_ratios(model_hash)

    # キャッシュデータが見つからない場合の処理
    if mag_ratios is None:
        status_text = f"Cache Inactive: Data not found"
        print(f"[MagCache Apply] Warning: Calibration data for '{model_name}' (hash: {model_hash}) not found. MagCache is disabled.")
        return {"ui": {"text": [status_text]}, "result": (new_model,)}
    
    # キャッシュデータが正常に読み込めた場合の処理
    status_text = f"Cache Active: {model_hash}.json"
    
    diffusion_model.magcache_original_forward = diffusion_model.forward
    diffusion_model.forward = universal_magcache_forward.__get__(diffusion_model, type(diffusion_model))
    diffusion_model.magcache_state = MagCacheState()
    diffusion_model.magcache_is_calibrating = False

    def unet_wrapper_function(model_function, kwargs):
        sigmas_tensor = kwargs['c']['transformer_options'].get('sample_sigmas')
        if sigmas_tensor is None: return model_function(kwargs['input'], kwargs['timestep'], **kwargs['c'])

        current_run_id = id(sigmas_tensor)
        if getattr(diffusion_model, 'magcache_last_run_id', None) != current_run_id:
            diffusion_model.magcache_last_run_id = current_run_id
            diffusion_model.magcache_state.reset()
            
            total_steps = len(sigmas_tensor) - 1 if len(sigmas_tensor) > 1 else 1
            interpolated_ratios = interpolate_mag_ratios(mag_ratios, total_steps)
            
            if interpolated_ratios is not None:
                diffusion_model.magcache_params = {
                    'mag_ratios': interpolated_ratios, 
                    'delta_threshold': magcache_thresh, 
                    'K_skips': magcache_k, 
                    'start_step_abs': int(total_steps * retention_ratio), 
                    'end_step_abs': total_steps - 1 if exclude_last_step else total_steps
                }
                print(f"[MagCache Apply] Parameters set for {total_steps} steps. Caching enabled from step {diffusion_model.magcache_params['start_step_abs']}.")
            else:
                if hasattr(diffusion_model, 'magcache_params'): delattr(diffusion_model, 'magcache_params')

        current_sigma = kwargs.get('timestep')[0]
        comparison = torch.isclose(sigmas_tensor, current_sigma)
        indices = torch.where(comparison)[0]
        diffusion_model.magcache_current_step = indices[0].item() if len(indices) > 0 else 0
        
        return model_function(kwargs['input'], kwargs['timestep'], **kwargs['c'])
    
    new_model.set_model_unet_function_wrapper(unet_wrapper_function)
    print(f"[MagCache Apply] Enabled and patched. Status: {status_text}")
    
    return {"ui": {"text": [status_text]}, "result": (new_model,)}


# --- ノードクラス定義 ---

class MagCacheCalibrate(object):
    @classmethod
    def INPUT_TYPES(s): return {"required": {"model": ("MODEL",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_calibration"
    CATEGORY = "MagCache"

    def apply_calibration(self, model):
        model_name = _get_model_name(model)
        if not model_name: return (model,)
        diffusion_model = model.get_model_object("diffusion_model")
        
        def unet_wrapper_function(model_function, kwargs):
            sigmas_tensor = kwargs['c']['transformer_options'].get('sample_sigmas')
            if sigmas_tensor is None: return model_function(kwargs['input'], kwargs['timestep'], **kwargs['c'])
            
            current_run_id = id(sigmas_tensor)
            last_run_id = getattr(diffusion_model, 'magcache_last_run_id', None)

            if current_run_id != last_run_id:
                if last_run_id is not None: _final_save_on_exit()
                print("-" * 40)
                print(f"[MagCache Calibrate] Initializing for new run (ID: {current_run_id})")
                print(f"[MagCache Calibrate] Model: {model_name}")
                print("-" * 40)
                diffusion_model.magcache_calibration_data = {'cond_ref': [], 'uncond_ref': [], 'last_cond_ref_eps': None, 'cond_noref': [], 'uncond_noref': [], 'last_cond_noref_eps': None}
                diffusion_model.magcache_last_run_id = current_run_id
                diffusion_model.magcache_model_name = model_name
                diffusion_model.magcache_is_calibrating = True
                _atexit_data_store['model_ref'] = weakref.ref(diffusion_model)

            computed_output = model_function(kwargs['input'], kwargs['timestep'], **kwargs['c'])
            _perform_calibration_logic(diffusion_model, computed_output, **kwargs)
            return computed_output

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)

class MagCacheApply(object):
    @classmethod
    def INPUT_TYPES(s): 
        return { 
            "required": { 
                "model": ("MODEL",), 
                "enabled": ("BOOLEAN", {"default": True}), 
                "magcache_thresh": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}), 
                "magcache_k": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}), 
                "retention_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}), 
                "exclude_last_step": ("BOOLEAN", {"default": True}), 
            } 
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_magcache"
    CATEGORY = "MagCache"

    def apply_magcache(self, model, enabled, magcache_thresh, magcache_k, retention_ratio, exclude_last_step):
        return _apply_magcache_logic(self, model, enabled, magcache_thresh, magcache_k, retention_ratio, exclude_last_step)

# --- [新規] 初心者向けの簡易適用ノード ---
class MagCacheEasyApply(object):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "exclude_last_step": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_magcache_easy"
    CATEGORY = "MagCache"

    def apply_magcache_easy(self, model, enabled, exclude_last_step):
        # 推奨デフォルト値をハードコード
        magcache_thresh = 0.1
        magcache_k = 2
        retention_ratio = 0.2
        return _apply_magcache_logic(model, enabled, magcache_thresh, magcache_k, retention_ratio, exclude_last_step)

# --- ノードのマッピング情報 ---
# __init__.pyで最終的に上書きされるが、念のためこちらも更新
NODE_CLASS_MAPPINGS = {
    "MagCacheApply": MagCacheApply,
    "MagCacheEasyApply": MagCacheEasyApply,
    "MagCacheCalibrate": MagCacheCalibrate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MagCacheApply": "MagCache Apply (Advanced)",
    "MagCacheEasyApply": "MagCache Easy Apply",
    "MagCacheCalibrate": "MagCache Calibrate",
}
