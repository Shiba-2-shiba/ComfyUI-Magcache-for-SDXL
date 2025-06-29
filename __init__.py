# __init__.py (パッチ適用機能付き・UNETLoader対応)

import nodes
import comfy.sd
from .magcache_nodes import MagCacheApply, MagCacheCalibrate

# --- ここからモンキーパッチ ---

# === CheckpointLoaderSimpleのパッチ (既存のSDXL用) ===
try:
    original_load_checkpoint = nodes.CheckpointLoaderSimple.load_checkpoint

    def patched_load_checkpoint(self, ckpt_name):
        model, clip, vae = original_load_checkpoint(self, ckpt_name)
        if model is not None:
            setattr(model, "magcache_source_ckpt_name", ckpt_name)
            print(f"[MagCache Patcher] Attached ckpt name '{ckpt_name}' to model from CheckpointLoaderSimple.")
        return (model, clip, vae)

    nodes.CheckpointLoaderSimple.load_checkpoint = patched_load_checkpoint
    print("[MagCache] Patched CheckpointLoaderSimple to inject model name.")
except Exception as e:
    print(f"[MagCache] Failed to patch CheckpointLoaderSimple: {e}")


# === UNETLoaderのパッチ (OmniGen2などdiffusion model用) ===
try:
    if hasattr(nodes, 'UNETLoader'):
        original_load_unet = nodes.UNETLoader.load_unet

        def patched_load_unet(self, unet_name, weight_dtype):
            # オリジナルのローダーを呼び出す (返り値はタプル)
            model, = original_load_unet(self, unet_name, weight_dtype)
            
            # modelオブジェクトにファイル名を属性として追加
            if model is not None:
                setattr(model, "magcache_source_ckpt_name", unet_name)
                print(f"[MagCache Patcher] Attached unet name '{unet_name}' to model from UNETLoader.")
            
            return (model,)

        nodes.UNETLoader.load_unet = patched_load_unet
        print("[MagCache] Patched UNETLoader to inject model name.")
    else:
        print("[MagCache] Warning: UNETLoader not found in nodes.py, skipping patch for it.")
except Exception as e:
    print(f"[MagCache] Failed to patch UNETLoader: {e}")

# --- ここまでモンキーパッチ ---


# ノード登録処理
NODE_CLASS_MAPPINGS = {
    "MagCacheApply": MagCacheApply,
    "MagCacheCalibrate": MagCacheCalibrate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagCacheApply": "MagCache Apply (SDXL/OmniGen2)",
    "MagCacheCalibrate": "MagCache Calibrate (SDXL/OmniGen2)",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### MagCache for SDXL/OmniGen2 loaded ###")
