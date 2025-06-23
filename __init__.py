# __init__.py (パッチ適用機能付き)

import nodes
import comfy.sd
from .magcache_nodes import MagCacheSDXL, MagCacheSDXLCalibration

# --- ここからモンキーパッチ ---
# オリジナルの関数をバックアップ
original_load_checkpoint = nodes.CheckpointLoaderSimple.load_checkpoint

def patched_load_checkpoint(self, ckpt_name):
    # オリジナルのローダー関数を呼び出して、model, clip, vaeを取得
    model, clip, vae = original_load_checkpoint(self, ckpt_name)
    
    # modelオブジェクト (ModelPatcher) に、使用されたckpt_nameを属性として追加
    if model is not None:
        # 他のノードと競合しないように、ユニークな属性名を付ける
        setattr(model, "magcache_source_ckpt_name", ckpt_name)
        print(f"[MagCache-SDXL Patcher] Attached checkpoint name '{ckpt_name}' to model object.")
    
    return (model, clip, vae)

# CheckpointLoaderSimpleのload_checkpointを、パッチを当てたバージョンに差し替える
nodes.CheckpointLoaderSimple.load_checkpoint = patched_load_checkpoint

print("### MagCache for SDXL: CheckpointLoaderSimple patched to inject model name. ###")
# --- ここまでモンキーパッチ ---


# 元のノード登録処理
NODE_CLASS_MAPPINGS = {
    "MagCacheSDXL": MagCacheSDXL,
    "MagCacheSDXLCalibration": MagCacheSDXLCalibration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagCacheSDXL": "MagCache for SDXL",
    "MagCacheSDXLCalibration": "Calibrate MagCache for SDXL",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### MagCache for SDXL loaded ###")