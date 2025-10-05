# __init__.py (リファクタリング版)

import nodes
import comfy.sd
# EasyApplyノードをインポートリストに追加
from .magcache_nodes import MagCacheApply, MagCacheEasyApply, MagCacheCalibrate

# --- ここからモンキーパッチ (変更なし) ---

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
            model, = original_load_unet(self, unet_name, weight_dtype)
            
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


# --- [変更点] ノード登録処理 ---
NODE_CLASS_MAPPINGS = {
    "MagCacheApply": MagCacheApply,
    "MagCacheEasyApply": MagCacheEasyApply,  # EasyApplyを追加
    "MagCacheCalibrate": MagCacheCalibrate,
}

# --- [変更点] ノード表示名の定義 ---
NODE_DISPLAY_NAME_MAPPINGS = {
    "MagCacheApply": "MagCache Apply (Advanced)",  # Advanced設定用であることを明記
    "MagCacheEasyApply": "MagCache Easy Apply",      # EasyApplyの表示名を設定
    "MagCacheCalibrate": "MagCache Calibrate (SDXL/OmniGen2)", # こちらは変更なし
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### MagCache for SDXL/OmniGen2 (UX Improved) loaded ###")
