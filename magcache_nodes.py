# magcache_nodes.py (モデル判別ロジック修正版)
#
# SDXLとOmniGen2の両モデルに対応するよう改修。
# モデルオブジェクトのクラス名を直接確認することで、より確実にモデルタイプを判別します。

import torch
import numpy as np

# --- コア機能とヘルパーをインポート ---
from .magcache_core import (
    get_model_hash, load_mag_ratios, save_mag_ratios,
    interpolate_mag_ratios, MagCacheState
)

# --- ヘルパー関数 ---

def _get_model_name(model):
    """
    パッチされたローダーによってmodelオブジェクトに注入されたチェックポイント名を取得する。
    """
    model_name = getattr(model, "magcache_source_ckpt_name", None)
    if model_name:
        print(f"[MagCache] Successfully retrieved model name via patch: {model_name}")
        return model_name
    else:
        print("[MagCache] Warning: Could not find patched model name.")
        print("[MagCache] This can happen if a custom loader node is used that isn't supported by the patch.")
        print("[MagCache] MagCache will be disabled for this workflow.")
        return None

def _get_model_type(model):
    """
    ModelPatcherオブジェクトからモデルの種別を判定する。
    モデルオブジェクトのクラス名を直接チェックして、OmniGen2かSDXL系かを識別する。
    """
    try:
        # ModelPatcherから実際のモデルオブジェクトを取得
        diffusion_model_wrapper = model.get_model_object("diffusion_model")
        
        # モデルオブジェクトのクラス名を取得
        model_class_name = diffusion_model_wrapper.__class__.__name__
        print(f"[MagCache] Detected model class: {model_class_name}")

        # クラス名に 'Omnigen2' が含まれているかで判定
        if 'Omnigen2' in model_class_name:
            print("[MagCache] Model type identified as: OmniGen2")
            return 'OmniGen2'
        else:
            # それ以外はSDXL系として扱う
            print("[MagCache] Model type identified as: SDXL")
            return 'SDXL'
            
    except Exception as e:
        print(f"[MagCache] Failed to determine model type by class name: {e}. Falling back to SDXL.")

    return 'SDXL'

# --- グローバル変数 ---
calibration_data = {
    'cond': [], 'uncond': [],
    'last_cond_eps': None, 'last_uncond_eps': None
}

def reset_calibration_data():
    global calibration_data
    calibration_data = { 'cond': [], 'uncond': [], 'last_cond_eps': None, 'last_uncond_eps': None }

# --- 共通のキャリブレーションロジック ---
def _perform_calibration_logic(computed_output, **kwargs):
    """Calibrates by calculating ratios between consecutive steps."""
    try:
        transformer_options = kwargs.get('transformer_options', {})
        cond_or_uncond = transformer_options.get('cond_or_uncond', [0] * computed_output.shape[0])
        
        cond_indices = [i for i, v in enumerate(cond_or_uncond) if v == 0]
        uncond_indices = [i for i, v in enumerate(cond_or_uncond) if v == 1]
        
        if cond_indices:
            current_cond_eps = computed_output[cond_indices].mean(dim=0, keepdim=True).detach()
            if calibration_data['last_cond_eps'] is not None:
                ratio = torch.linalg.norm(current_cond_eps) / (torch.linalg.norm(calibration_data['last_cond_eps']) + 1e-9)
                calibration_data['cond'].append(ratio.item())
            calibration_data['last_cond_eps'] = current_cond_eps
            
        if uncond_indices:
            current_uncond_eps = computed_output[uncond_indices].mean(dim=0, keepdim=True).detach()
            if calibration_data['last_uncond_eps'] is not None:
                ratio = torch.linalg.norm(current_uncond_eps) / (torch.linalg.norm(calibration_data['last_uncond_eps']) + 1e-9)
                calibration_data['uncond'].append(ratio.item())
            calibration_data['last_uncond_eps'] = current_uncond_eps
    except Exception as e:
        print(f"[MagCache] Calibration logic error: {e}")

# --- パッチ用のフォワード関数 (SDXL用) ---
def sdxl_calibration_forward(self, x, timesteps, context, **kwargs):
    computed_output = self.magcache_original_forward(x, timesteps, context, **kwargs)
    _perform_calibration_logic(computed_output, **kwargs)
    return computed_output

def sdxl_magcache_forward(self, x, timesteps, context, **kwargs):
    return _magcache_forward(self, self.magcache_original_forward, x, timesteps, context=context, **kwargs)

# --- パッチ用のフォワード関数 (OmniGen2用) ---
def omnigen2_calibration_forward(self, x, timesteps, context, num_tokens, ref_latents=None, attention_mask=None, **kwargs):
    computed_output = self.magcache_original_forward(x, timesteps, context, num_tokens, ref_latents, attention_mask, **kwargs)
    _perform_calibration_logic(computed_output, **kwargs)
    return computed_output

def omnigen2_magcache_forward(self, x, timesteps, context, num_tokens, ref_latents=None, attention_mask=None, **kwargs):
    return _magcache_forward(self, self.magcache_original_forward, x, timesteps, context=context, num_tokens=num_tokens, ref_latents=ref_latents, attention_mask=attention_mask, **kwargs)

# --- 共通のMagCache推論ロジック ---
def _magcache_forward(self, original_forward, x, timesteps, **kwargs_for_forward):
    params = getattr(self, 'magcache_params', None)
    state_manager = getattr(self, 'magcache_state', None)
    if not params or not state_manager:
        return original_forward(x, timesteps, **kwargs_for_forward)

    current_step = getattr(self, 'magcache_current_step', 0)
    cond_or_uncond = kwargs_for_forward.get('transformer_options', {}).get('cond_or_uncond', [0] * x.shape[0])
    
    skip_this_step = False
    if current_step >= params['start_step_abs'] and current_step < params['end_step_abs']:
        def check_skip(guidance_type, ratio_offset):
            state = state_manager.get_state(guidance_type)
            ratio_index = current_step * 2 + ratio_offset
            if ratio_index >= len(params['mag_ratios']) or state['residual_cache'] is None: return False
            ratio = params['mag_ratios'][ratio_index]
            state['accumulated_ratio'] *= ratio
            state['accumulated_steps'] += 1
            state['accumulated_err'] += abs(1.0 - state['accumulated_ratio'])
            return (state['accumulated_err'] < params['delta_threshold'] and state['accumulated_steps'] <= params['K_skips'])
            
        can_skip_cond = True if 0 not in cond_or_uncond else check_skip("cond", 0)
        can_skip_uncond = True if 1 not in cond_or_uncond else check_skip("uncond", 1)
        if can_skip_cond and can_skip_uncond: skip_this_step = True
        
    if skip_this_step:
        print(f"[MagCache] Step {current_step}: Cache HIT -> Skipping UNet call.")
        output = torch.zeros_like(x)
        if 0 in cond_or_uncond:
            output[[i for i, v in enumerate(cond_or_uncond) if v == 0]] = state_manager.get_state('cond')['residual_cache'].to(x.device, x.dtype)
        if 1 in cond_or_uncond:
             output[[i for i, v in enumerate(cond_or_uncond) if v == 1]] = state_manager.get_state('uncond')['residual_cache'].to(x.device, x.dtype)
        return output
    else:
        def reset_state(guidance_type):
            state = state_manager.get_state(guidance_type)
            state.update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
        if 0 in cond_or_uncond: reset_state("cond")
        if 1 in cond_or_uncond: reset_state("uncond")
        
        computed_output = original_forward(x, timesteps, **kwargs_for_forward)
        
        cond_indices = [i for i, v in enumerate(cond_or_uncond) if v == 0]
        uncond_indices = [i for i, v in enumerate(cond_or_uncond) if v == 1]
        if cond_indices: state_manager.store_residual(computed_output[cond_indices].mean(dim=0, keepdim=True).detach(), "cond")
        if uncond_indices: state_manager.store_residual(computed_output[uncond_indices].mean(dim=0, keepdim=True).detach(), "uncond")
        return computed_output

# --- ノードクラス定義 ---

class MagCacheCalibrate:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "model": ("MODEL",) } }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_calibration"
    CATEGORY = "MagCache"

    def apply_calibration(self, model):
        model_name = _get_model_name(model)
        if not model_name: return (model,)

        model_type = _get_model_type(model)
        diffusion_model = model.get_model_object("diffusion_model")

        if not hasattr(diffusion_model, 'magcache_original_forward'):
            diffusion_model.magcache_original_forward = diffusion_model.forward

        if model_type == 'OmniGen2':
            diffusion_model.forward = omnigen2_calibration_forward.__get__(diffusion_model, type(diffusion_model))
        else:
            diffusion_model.forward = sdxl_calibration_forward.__get__(diffusion_model, type(diffusion_model))
        
        def unet_wrapper_function(model_function, kwargs):
            c_kwargs = kwargs.get('c', {})
            all_sigmas = c_kwargs.get('transformer_options', {}).get('sample_sigmas')
            current_sigma = kwargs.get('timestep')[0]

            if all_sigmas is not None:
                comparison = torch.isclose(all_sigmas, current_sigma)
                indices = torch.where(comparison)[0]
                diffusion_model.magcache_current_step = indices[0].item() if len(indices) > 0 else 0
                diffusion_model.magcache_total_steps = len(all_sigmas) -1 if len(all_sigmas) > 1 else len(all_sigmas)
            else:
                diffusion_model.magcache_current_step = 0
                diffusion_model.magcache_total_steps = 1

            if diffusion_model.magcache_current_step == 0:
                reset_calibration_data()
            
            output = model_function(kwargs['input'], kwargs['timestep'], **kwargs['c'])
            
            if diffusion_model.magcache_current_step == diffusion_model.magcache_total_steps - 1:
                cond, uncond = calibration_data['cond'], calibration_data['uncond']
                min_len = min(len(cond), len(uncond))
                if min_len > 0:
                    interleaved = np.empty((min_len * 2,), dtype=np.float32)
                    interleaved[0::2] = cond[:min_len]
                    interleaved[1::2] = uncond[:min_len]
                    model_hash = get_model_hash(model_name)
                    save_mag_ratios(model_hash, interleaved.tolist(), model_name)
                
                if hasattr(diffusion_model, 'magcache_original_forward'):
                    diffusion_model.forward = diffusion_model.magcache_original_forward
                    delattr(diffusion_model, 'magcache_original_forward')
            return output

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)

class MagCacheApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "enabled": ("BOOLEAN", {"default": True}),
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
        if not enabled:
            diffusion_model = model.get_model_object("diffusion_model")
            if hasattr(diffusion_model, 'magcache_original_forward'):
                diffusion_model.forward = diffusion_model.magcache_original_forward
                delattr(diffusion_model, 'magcache_original_forward')
            new_model = model.clone()
            new_model.set_model_unet_function_wrapper(None)
            return (new_model,)

        model_name = _get_model_name(model)
        if not model_name: return (model,)
        
        model_hash = get_model_hash(model_name)
        mag_ratios = load_mag_ratios(model_hash)
        
        if mag_ratios is None:
            print(f"[MagCache] Warning: Calibration data for '{model_name}' not found. MagCache is disabled.")
            return (model,)
        
        model_type = _get_model_type(model)
        diffusion_model = model.get_model_object("diffusion_model")

        if not hasattr(diffusion_model, 'magcache_original_forward'):
            diffusion_model.magcache_original_forward = diffusion_model.forward
        
        if model_type == 'OmniGen2':
            diffusion_model.forward = omnigen2_magcache_forward.__get__(diffusion_model, type(diffusion_model))
        else:
            diffusion_model.forward = sdxl_magcache_forward.__get__(diffusion_model, type(diffusion_model))

        if not hasattr(diffusion_model, 'magcache_state'):
            diffusion_model.magcache_state = MagCacheState()

        def unet_wrapper_function(model_function, kwargs):
            c_kwargs = kwargs.get('c', {})
            all_sigmas = c_kwargs.get('transformer_options', {}).get('sample_sigmas')
            current_sigma = kwargs.get('timestep')[0]

            if all_sigmas is not None:
                comparison = torch.isclose(all_sigmas, current_sigma)
                indices = torch.where(comparison)[0]
                diffusion_model.magcache_current_step = indices[0].item() if len(indices) > 0 else 0
                diffusion_model.magcache_total_steps = len(all_sigmas) -1 if len(all_sigmas) > 1 else len(all_sigmas)
            else:
                diffusion_model.magcache_current_step = 0
                diffusion_model.magcache_total_steps = 1
            
            if diffusion_model.magcache_current_step == 0:
                state = diffusion_model.magcache_state
                state.reset()
                
                interpolated_ratios = interpolate_mag_ratios(mag_ratios, diffusion_model.magcache_total_steps)
                
                if interpolated_ratios is not None:
                    diffusion_model.magcache_params = {
                        'mag_ratios': interpolated_ratios, 
                        'delta_threshold': magcache_thresh, 
                        'K_skips': magcache_k,
                        'start_step_abs': int(diffusion_model.magcache_total_steps * retention_ratio),
                        'end_step_abs': diffusion_model.magcache_total_steps - 1 if exclude_last_step else diffusion_model.magcache_total_steps
                    }
                else:
                    print(f"[MagCache] Warning: Ratio interpolation failed. Disabling cache for this run.")
                    if hasattr(diffusion_model, 'magcache_params'):
                        delattr(diffusion_model, 'magcache_params')

            return model_function(kwargs['input'], kwargs['timestep'], **kwargs['c'])
        
        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)

# --- ノードのマッピング情報 ---
NODE_CLASS_MAPPINGS = {
    "MagCacheApply": MagCacheApply,
    "MagCacheCalibrate": MagCacheCalibrate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagCacheApply": "MagCache Apply (SDXL/OmniGen2)",
    "MagCacheCalibrate": "MagCache Calibrate (SDXL/OmniGen2)",
}
