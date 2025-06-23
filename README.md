# ComfyUI-Magcache-for-SDXL

Magcache implementation for SDXL

## how to install

```
Your directory /ComfyUI/ComfyUI\custom_nodes/

Git clone https://github.com/Shiba-2-shiba/ComfyUI-Magcache-for-SDXL.git

```
## Caliblate MagCache for SDXL Node

![初期化画面](img/img1.png)

First you shold add this node in your workflow and queue the workflow.

After generation, json file is in the magcache_data folder.

1json file is generated in 1 model. 

If you use other sampler in same model, I suggest to delete the json file and regeneration new json file.

## MagCache for SDXL Node

![初期化画面](img/img2.png)

The you can add this node in your workflow. This node only work if the model comfirmed json file is in the magcache_data folder.

You can change parameter to adjust the image quality.
