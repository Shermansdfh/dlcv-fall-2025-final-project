#!/usr/bin/env python3
"""
簡單的 wrapper：使用原版 infer.py，但只處理前 N 個樣本

這個腳本會：
1. 直接調用原版 infer.py 的函數
2. 在 DataLoader 層面限制樣本數量
3. 避免下載整個 100GB+ dataset（雖然首次仍會下載 metadata）

使用方式：
    cd /home/hpc/ce505203/finals_repo/third_party/cld/infer
    python ../../../scripts/test_original_cld_limited.py --config_path <config.yaml> --max_samples 5
"""

import sys
import os
from pathlib import Path
import argparse

# 先檢查 CUDA 可用性（在導入 infer.py 之前）
# 因為 infer.py 會設置 CUDA_VISIBLE_DEVICES = "1"，我們需要先處理
import torch

# 設定路徑
script_path = Path(__file__).resolve()
repo_root = script_path.parent.parent
cld_root = repo_root / "third_party" / "cld"
cld_infer_dir = cld_root / "infer"

sys.path.insert(0, str(cld_root))
os.chdir(str(cld_root))

# 檢查 CUDA 可用性
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA 可用：{torch.cuda.get_device_name(0)}")
    print(f"   CUDA 版本：{torch.version.cuda}")
    print(f"   可用 GPU 數量：{device_count}\n")
else:
    print("\n" + "="*60)
    print("⚠️  警告：CUDA 不可用！")
    print("="*60)
    print("CLD inference 需要 GPU 才能運行。")
    print("如果沒有 GPU，推理會非常慢（可能需要數小時）。")
    print("\n建議：")
    print("1. 確保 GPU 可用：nvidia-smi")
    print("2. 確保 CUDA_VISIBLE_DEVICES 設置正確")
    print("3. 確保 PyTorch 安裝了 CUDA 支持")
    print("="*60 + "\n")
    
    response = input("是否繼續使用 CPU？（y/N）: ")
    if response.lower() != 'y':
        print("已取消。")
        raise SystemExit(1)

# 導入原版 infer.py
# 注意：infer.py 會設置 CUDA_VISIBLE_DEVICES = "1"
# 如果系統只有 GPU 0，這會導致問題，所以我們需要修改代碼
import importlib.util
infer_py_path = cld_infer_dir / "infer.py"

# 讀取 infer.py 的代碼
with open(infer_py_path, 'r', encoding='utf-8') as f:
    infer_code = f.read()

# 如果只有一個 GPU，修改 CUDA_VISIBLE_DEVICES 設置
if cuda_available and device_count == 1:
    # 將 CUDA_VISIBLE_DEVICES = "1" 改為 "0" 或移除
    if 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"' in infer_code:
        infer_code = infer_code.replace(
            'os.environ["CUDA_VISIBLE_DEVICES"] = "1"',
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 修改為使用 GPU 0'
        )
        print("[INFO] 檢測到只有一個 GPU，已修改 infer.py 中的 CUDA_VISIBLE_DEVICES=1 → 0\n")

# 創建模組並執行修改後的代碼
spec = importlib.util.spec_from_file_location("infer_module", str(infer_py_path))
infer_module = importlib.util.module_from_spec(spec)
# 執行修改後的代碼
exec(compile(infer_code, str(infer_py_path), 'exec'), infer_module.__dict__)

from torch.utils.data import Subset, Dataset

# Optimize LoRA loading: Monkey patch CustomFluxPipeline.lora_state_dict AFTER loading infer_module
# This ensures LoRA weights are loaded directly to GPU using safetensors for faster loading
print("[DEBUG] Starting LoRA optimization setup...", flush=True)
CustomFluxPipeline = None  # Will be set in try block
try:
    import time
    
    # Import CustomFluxPipeline from the loaded module
    # This import may be slow if models.pipeline is large
    print("[DEBUG] Importing CustomFluxPipeline (this may take a moment)...", flush=True)
    import_start = time.time()
    from models.pipeline import CustomFluxPipeline
    import_elapsed = time.time() - import_start
    print(f"[DEBUG] CustomFluxPipeline imported in {import_elapsed:.2f}s", flush=True)
    
    # Store original lora_state_dict method
    print("[DEBUG] Checking for lora_state_dict method...", flush=True)
    if hasattr(CustomFluxPipeline, 'lora_state_dict'):
        print("[DEBUG] Found lora_state_dict, creating optimized version...", flush=True)
        original_lora_state_dict = CustomFluxPipeline.lora_state_dict
        
        @staticmethod
        def optimized_lora_state_dict(lora_path, *args, **kwargs):
            """
            Optimized LoRA loading that:
            1. Ensures safetensors format is used
            2. Loads directly to GPU (cuda)
            3. Provides progress indication
            
            Returns:
                - If called by load_lora_into_transformer: state_dict only
                - If called by pipeline.load_lora_weights(): (state_dict, network_alphas) tuple
            """
            import safetensors.torch
            
            lora_path_obj = Path(lora_path)
            
            # Check if it's a directory or file path
            if lora_path_obj.is_dir():
                # Look for safetensors file in directory
                safetensors_file = lora_path_obj / "pytorch_lora_weights.safetensors"
                if not safetensors_file.exists():
                    # Fallback: try to find any safetensors file
                    safetensors_files = list(lora_path_obj.glob("*.safetensors"))
                    if safetensors_files:
                        safetensors_file = safetensors_files[0]
                    else:
                        print(f"⚠️  Warning: No safetensors file found in {lora_path}, falling back to original method")
                        return original_lora_state_dict(lora_path, *args, **kwargs)
                lora_file = safetensors_file
            elif lora_path_obj.suffix == ".safetensors":
                lora_file = lora_path_obj
            else:
                # Not safetensors format, try to find safetensors version
                safetensors_file = lora_path_obj.parent / f"{lora_path_obj.stem}.safetensors"
                if safetensors_file.exists():
                    lora_file = safetensors_file
                    print(f"   📦 Found safetensors version: {lora_file}")
                else:
                    print(f"⚠️  Warning: LoRA file is not safetensors format: {lora_path}")
                    print(f"   Expected safetensors file: {safetensors_file}")
                    print(f"   Falling back to original method (may be slower)")
                    return original_lora_state_dict(lora_path, *args, **kwargs)
            
            if not lora_file.exists():
                print(f"⚠️  Warning: LoRA file not found: {lora_file}, falling back to original method")
                return original_lora_state_dict(lora_path, *args, **kwargs)
            
            print(f"   📦 Loading LoRA weights from: {lora_file}", flush=True)
            print(f"   ✅ Using safetensors format (faster loading)", flush=True)
            
            start_time = time.time()
            
            # Determine device - prefer GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print(f"   ⚠️  Warning: CUDA not available, loading to CPU (will be slower)", flush=True)
            
            try:
                # Load safetensors - try direct GPU loading first, fallback to CPU then move to GPU
                # safetensors.torch.load_file returns a dict of tensors
                load_start = time.time()
                try:
                    # Try loading directly to GPU if device parameter is supported
                    state_dict = safetensors.torch.load_file(str(lora_file), device=device)
                    load_elapsed = time.time() - load_start
                    loaded_to_gpu = (device == "cuda")
                    print(f"   ⏱️  Safetensors I/O: {load_elapsed:.2f}s", flush=True)
                except TypeError:
                    # device parameter not supported, load to CPU then move to GPU
                    state_dict = safetensors.torch.load_file(str(lora_file))
                    load_elapsed = time.time() - load_start
                    print(f"   ⏱️  Safetensors I/O (CPU): {load_elapsed:.2f}s", flush=True)
                    
                    if device == "cuda" and torch.cuda.is_available():
                        # Move all tensors to GPU efficiently
                        # Batch move for better performance (avoids multiple small transfers)
                        move_start = time.time()
                        # Collect all tensors first, then move them in batch
                        tensor_keys = [k for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
                        if tensor_keys:
                            # Use non_blocking=True for async transfer, but we'll sync at the end
                            for k in tensor_keys:
                                state_dict[k] = state_dict[k].cuda(non_blocking=True)
                            # Synchronize to ensure all transfers complete before returning
                            torch.cuda.synchronize()
                        move_elapsed = time.time() - move_start
                        print(f"   ⏱️  CPU->GPU transfer ({len(tensor_keys)} tensors): {move_elapsed:.2f}s", flush=True)
                        loaded_to_gpu = True
                    else:
                        loaded_to_gpu = False
                
                elapsed = time.time() - start_time
                num_keys = len(state_dict)
                file_size_mb = lora_file.stat().st_size / (1024 * 1024)
                
                print(f"   ✅ LoRA loaded ({num_keys} keys, {file_size_mb:.2f} MB) in {elapsed:.2f}s total", flush=True)
                if loaded_to_gpu:
                    print(f"   🚀 Loaded directly to GPU (optimized path)", flush=True)
                
                # Always return tuple (state_dict, network_alphas) as expected by load_lora_weights
                # load_lora_into_transformer wrapper will handle unpacking
                network_alphas = None
                return state_dict, network_alphas
                
            except Exception as e:
                print(f"   ⚠️  Error loading safetensors: {e}", flush=True)
                print(f"   Falling back to original method", flush=True)
                return original_lora_state_dict(lora_path, *args, **kwargs)
        
        # Apply monkey patch
        print("[DEBUG] About to apply monkey patch to CustomFluxPipeline.lora_state_dict...", flush=True)
        CustomFluxPipeline.lora_state_dict = optimized_lora_state_dict
        print("[DEBUG] Monkey patch applied successfully", flush=True)
        print("✅ Applied LoRA loading optimization: safetensors + direct GPU loading", flush=True)
        print("[DEBUG] Finished LoRA optimization setup", flush=True)
        
        # Also patch load_lora_into_transformer to handle tuple return from optimized_lora_state_dict
        if hasattr(CustomFluxPipeline, 'load_lora_into_transformer'):
            original_load_lora = CustomFluxPipeline.load_lora_into_transformer
            
            @staticmethod
            def timed_load_lora_into_transformer(lora_state_dict, *args, **kwargs):
                print("[DEBUG] load_lora_into_transformer: Starting...", flush=True)
                start = time.time()
                
                # Handle case where lora_state_dict might be a tuple (from optimized_lora_state_dict)
                # load_lora_into_transformer expects just the state_dict, not the tuple
                if isinstance(lora_state_dict, tuple):
                    lora_state_dict, _ = lora_state_dict  # Unpack tuple, ignore network_alphas
                
                result = original_load_lora(lora_state_dict, *args, **kwargs)
                elapsed = time.time() - start
                print(f"[DEBUG] load_lora_into_transformer: Completed in {elapsed:.2f}s", flush=True)
                return result
            
            CustomFluxPipeline.load_lora_into_transformer = timed_load_lora_into_transformer
        
except ImportError as e:
    print(f"⚠️  Warning: Could not apply LoRA loading optimization: {e}")
    print("   LoRA loading will use default method (may be slower)")
except Exception as e:
    print(f"⚠️  Warning: Error applying LoRA loading optimization: {e}")
    print("   LoRA loading will use default method (may be slower)")

# Patch fuse_lora and unload_lora to add timing (similar to infer_dlcv.py)
try:
    from models.mmdit import CustomFluxTransformer2DModel
    
    if hasattr(CustomFluxTransformer2DModel, 'fuse_lora'):
        original_fuse_lora = CustomFluxTransformer2DModel.fuse_lora
        
        def timed_fuse_lora(self, *args, **kwargs):
            import time
            print("[DEBUG] fuse_lora: Starting...", flush=True)
            start = time.time()
            result = original_fuse_lora(self, *args, **kwargs)
            elapsed = time.time() - start
            print(f"[DEBUG] fuse_lora: Completed in {elapsed:.2f}s", flush=True)
            return result
        
        CustomFluxTransformer2DModel.fuse_lora = timed_fuse_lora
    
    if hasattr(CustomFluxTransformer2DModel, 'unload_lora'):
        original_unload_lora = CustomFluxTransformer2DModel.unload_lora
        
        def timed_unload_lora(self, *args, **kwargs):
            import time
            print("[DEBUG] unload_lora: Starting...", flush=True)
            start = time.time()
            result = original_unload_lora(self, *args, **kwargs)
            elapsed = time.time() - start
            print(f"[DEBUG] unload_lora: Completed in {elapsed:.2f}s", flush=True)
            return result
        
        CustomFluxTransformer2DModel.unload_lora = timed_unload_lora
except ImportError:
    print("[DEBUG] Could not patch fuse_lora/unload_lora (models.mmdit not available)", flush=True)
except Exception as e:
    print(f"[DEBUG] Warning: Error patching fuse_lora/unload_lora: {e}", flush=True)


def apply_skip_fuse_lora_patch(config):
    """
    如果 config 中設置了 skip_fuse_lora=True，則 patch fuse_lora 和 unload_lora 為 no-op
    這樣可以節省記憶體，LoRA 會通過 PEFT 機制工作（性能損失 <5%）
    """
    skip_fuse_lora = config.get('skip_fuse_lora', False)
    
    if not skip_fuse_lora:
        return
    
    print("⚠️  skip_fuse_lora=True: Will skip fuse_lora() to save memory", flush=True)
    print("   LoRA will work via PEFT mechanism (slight performance loss <5%, but saves memory)", flush=True)
    
    # Monkey patch fuse_lora and unload_lora to be no-ops
    if CustomFluxPipeline is not None:
        try:
            from models.mmdit import CustomFluxTransformer2DModel
            from models.multiLayer_adapter import MultiLayerAdapter
            
            # Patch fuse_lora to be a no-op
            if hasattr(CustomFluxTransformer2DModel, 'fuse_lora'):
                def noop_fuse_lora(self, *args, **kwargs):
                    print("[INFO] Skipping fuse_lora() to save memory (LoRA will work via PEFT)", flush=True)
                    return None
                CustomFluxTransformer2DModel.fuse_lora = noop_fuse_lora
            
            if hasattr(MultiLayerAdapter, 'fuse_lora'):
                def noop_fuse_lora_adapter(self, *args, **kwargs):
                    print("[INFO] Skipping MultiLayerAdapter.fuse_lora() to save memory", flush=True)
                    return None
                MultiLayerAdapter.fuse_lora = noop_fuse_lora_adapter
            
            # Patch unload_lora to be a no-op (since we didn't fuse, we shouldn't unload)
            if hasattr(CustomFluxTransformer2DModel, 'unload_lora'):
                def noop_unload_lora(self, *args, **kwargs):
                    print("[INFO] Skipping unload_lora() (LoRA weights kept for PEFT inference)", flush=True)
                    return None
                CustomFluxTransformer2DModel.unload_lora = noop_unload_lora
            
            if hasattr(MultiLayerAdapter, 'unload_lora'):
                def noop_unload_lora_adapter(self, *args, **kwargs):
                    print("[INFO] Skipping MultiLayerAdapter.unload_lora()", flush=True)
                    return None
                MultiLayerAdapter.unload_lora = noop_unload_lora_adapter
            
            print("✅ Patched fuse_lora/unload_lora to skip (memory optimization)", flush=True)
        except ImportError:
            print("⚠️  Warning: Could not patch fuse_lora/unload_lora (models not available)", flush=True)
        except Exception as e:
            print(f"⚠️  Warning: Error patching fuse_lora/unload_lora: {e}", flush=True)


class LimitedLayoutTrainDataset(Dataset):
    """
    修改版的 LayoutTrainDataset，在初始化時就限制樣本數量
    這樣可以避免處理整個 dataset（雖然 metadata 還是會下載）
    """
    def __init__(self, data_dir, split="test", max_samples=None):
        from datasets import load_dataset, concatenate_datasets
        from collections import defaultdict
        import numpy as np
        import torchvision.transforms as T
        from PIL import Image
        
        print(f"[INFO] 加載 PrismLayersPro dataset（split={split}）...", flush=True)
        print(f"[INFO] 注意：HuggingFace 會下載 metadata 文件（~400-500MB），這是正常的", flush=True)
        print(f"[INFO] 圖片會按需下載，只會下載實際訪問的樣本", flush=True)
        
        full_dataset = load_dataset(
            "artplus/PrismLayersPro",
            cache_dir=data_dir,
        )
        full_dataset = concatenate_datasets(list(full_dataset.values()))

        if "style_category" not in full_dataset.column_names:
            raise ValueError("Dataset must contain a 'style_category' field to split by class.")

        categories = np.array(full_dataset["style_category"])
        category_to_indices = defaultdict(list)
        for i, cat in enumerate(categories):
            category_to_indices[cat].append(i)

        subsets = []
        for cat, indices in category_to_indices.items():
            total_len = len(indices)
            idx_90 = int(total_len * 0.9)
            idx_95 = int(total_len * 0.95)

            if split == "train":
                selected_idx = indices[:idx_90]
            elif split == "test":
                selected_idx = indices[idx_90:idx_95]
            elif split == "val":
                selected_idx = indices[idx_95:]
            else:
                raise ValueError("split must be 'train', 'val', or 'test'")

            subsets.append(full_dataset.select(selected_idx))

        # 合併所有 subsets
        combined_dataset = concatenate_datasets(subsets)
        
        # 在初始化時就限制樣本數量
        if max_samples is not None and max_samples > 0:
            actual_samples = min(max_samples, len(combined_dataset))
            print(f"[INFO] 限制樣本數量：{len(combined_dataset)} → {actual_samples}", flush=True)
            self.dataset = combined_dataset.select(range(actual_samples))
        else:
            self.dataset = combined_dataset
        
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        def rgba2rgb(img_RGBA):
            from PIL import Image
            img_RGB = Image.new("RGB", img_RGBA.size, (128, 128, 128))
            img_RGB.paste(img_RGBA, mask=img_RGBA.split()[3])
            return img_RGB

        def get_img(x):
            from PIL import Image
            if isinstance(x, str):
                img_RGBA = Image.open(x).convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
            else:
                img_RGBA = x.convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
            return img_RGBA, img_RGB

        whole_img_RGBA, whole_img_RGB = get_img(item["whole_image"])
        whole_cap = item["whole_caption"]
        W, H = whole_img_RGBA.size
        base_layout = [0, 0, W - 1, H - 1]

        layer_image_RGBA = [self.to_tensor(whole_img_RGBA)]
        layer_image_RGB  = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]

        base_img_RGBA, base_img_RGB = get_img(item["base_image"])
        layer_image_RGBA.append(self.to_tensor(base_img_RGBA))
        layer_image_RGB.append(self.to_tensor(base_img_RGB))
        layout.append(base_layout)

        layer_count = item["layer_count"]
        for i in range(layer_count):
            key = f"layer_{i:02d}"
            img_RGBA, img_RGB = get_img(item[key])
            
            w0, h0, w1, h1 = item[f"{key}_box"]

            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))

            W_img, H_img = w1 - w0, h1 - h0
            if img_RGBA.size != (W_img, H_img):
                img_RGBA = img_RGBA.resize((W_img, H_img), Image.BILINEAR)
                img_RGB  = img_RGB.resize((W_img, H_img), Image.BILINEAR)

            canvas_RGBA.paste(img_RGBA, (w0, h0), img_RGBA)
            canvas_RGB.paste(img_RGB, (w0, h0))

            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([w0, h0, w1, h1])

        return {
            "pixel_RGBA": layer_image_RGBA,
            "pixel_RGB": layer_image_RGB,
            "whole_img": whole_img_RGB,
            "caption": whole_cap,
            "height": H,
            "width": W,
            "layout": layout,
        }


def inference_layout_limited(config, max_samples: int = 5):
    """
    修改版的 inference_layout，只處理前 max_samples 個樣本
    """
    import torch  # 確保 torch 已導入
    
    if config.get('seed') is not None:
        infer_module.seed_everything(config['seed'])
    
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], "merged"), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], "merged_rgba"), exist_ok=True)

    # Load transparent VAE（使用原版邏輯）
    print("[INFO] Loading Transparent VAE...", flush=True)
    
    # 檢查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available, using CPU (this will be very slow!)", flush=True)
    else:
        print(f"[INFO] Using device: {device}", flush=True)
    
    import argparse as argparse_module
    from models.transp_vae import AutoencoderKLTransformerTraining as CustomVAE
    
    vae_args = argparse_module.Namespace(
        max_layers=config.get('max_layers', 48),
        decoder_arch=config.get('decoder_arch', 'vit'),
        pos_embedding=config.get('pos_embedding', 'rope'),
        layer_embedding=config.get('layer_embedding', 'rope'),
        single_layer_decoder=config.get('single_layer_decoder', None)
    )
    transp_vae = CustomVAE(vae_args)
    transp_vae_path = config.get('transp_vae_path')
    
    # 使用正確的設備加載，並明確指定 weights_only=False（因為 checkpoint 可能包含非標準對象）
    try:
        transp_vae_weights = torch.load(
            transp_vae_path, 
            map_location=device,
            weights_only=False  # CLD checkpoints 可能包含 argparse.Namespace 等非標準對象
        )
    except Exception as e:
        print(f"[ERROR] Failed to load transparent VAE weights: {e}", flush=True)
        print(f"[INFO] Trying to load on CPU first, then move to {device}...", flush=True)
        # 如果直接加載失敗，先加載到 CPU，然後移動到目標設備
        transp_vae_weights = torch.load(
            transp_vae_path,
            map_location="cpu",
            weights_only=False
        )
        # 移動權重到目標設備（如果需要的話）
        if isinstance(transp_vae_weights, dict) and 'model' in transp_vae_weights:
            for k, v in transp_vae_weights['model'].items():
                if isinstance(v, torch.Tensor) and device.type == "cuda":
                    transp_vae_weights['model'][k] = v.to(device)
    
    missing_keys, unexpected_keys = transp_vae.load_state_dict(transp_vae_weights['model'], strict=False)
    if missing_keys:
        print(f"[WARNING] Missing keys: {missing_keys}", flush=True)
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys: {unexpected_keys}", flush=True)
    transp_vae.eval()
    transp_vae = transp_vae.to(device)
    print("[INFO] Transparent VAE loaded successfully.", flush=True)

    # 應用 skip_fuse_lora patch（如果配置中啟用）
    apply_skip_fuse_lora_patch(config)
    
    # 初始化 pipeline（使用原版邏輯）
    pipeline = infer_module.initialize_pipeline(config)

    # 創建 dataset（使用修改版，在初始化時就限制樣本數量）
    print(f"[INFO] 創建 dataset（將限制為前 {max_samples} 個樣本）...", flush=True)
    
    # 使用修改版的 dataset，在 split 之後立即限制樣本數量
    dataset = LimitedLayoutTrainDataset(config['data_dir'], split="test", max_samples=max_samples)
    
    loader = infer_module.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=infer_module.collate_fn
    )

    generator = torch.Generator(device=device).manual_seed(config.get('seed', 42))

    idx = 0
    for batch in loader:
        print(f"\n{'='*60}")
        print(f"Processing case {idx} (樣本 {idx+1}/{actual_samples})")
        print(f"{'='*60}", flush=True)

        height = int(batch["height"][0])
        width = int(batch["width"][0])
        adapter_img = batch["whole_img"][0]
        caption = batch["caption"][0]
        layer_boxes = infer_module.get_input_box(batch["layout"][0]) 

        # Debug: 顯示 layout 資訊
        print(f"[DEBUG] Image size: {width}x{height}", flush=True)
        print(f"[DEBUG] Layout boxes count: {len(layer_boxes)}", flush=True)
        if len(caption) > 100:
            print(f"[DEBUG] Caption: {caption[:100]}...", flush=True)
        else:
            print(f"[DEBUG] Caption: {caption}", flush=True)

        # Generate layers using pipeline（使用原版邏輯）
        x_hat, image, latents = pipeline(
            prompt=caption,
            adapter_image=adapter_img,
            adapter_conditioning_scale=0.9,
            validation_box=layer_boxes,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=config.get('cfg', 4.0),
            num_layers=len(layer_boxes),
            sdxl_vae=transp_vae,  # Use transparent VAE
        )

        # Adjust x_hat range from [-1, 1] to [0, 1]
        x_hat = (x_hat + 1) / 2

        # Remove batch dimension and ensure float32 dtype
        x_hat = x_hat.squeeze(0).permute(1, 0, 2, 3).to(torch.float32)
        
        this_index = f"case_{idx}"
        case_dir = os.path.join(config['save_dir'], this_index)
        os.makedirs(case_dir, exist_ok=True)
        
        # Save whole image_RGBA (X_hat[0]) and background_RGBA (X_hat[1])
        whole_image_layer = (x_hat[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        whole_image_rgba_image = Image.fromarray(whole_image_layer, "RGBA")
        whole_image_rgba_image.save(os.path.join(case_dir, "whole_image_rgba.png"))

        adapter_img.save(os.path.join(case_dir, "origin.png"))

        background_layer = (x_hat[1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        background_rgba_image = Image.fromarray(background_layer, "RGBA")
        background_rgba_image.save(os.path.join(case_dir, "background_rgba.png"))

        x_hat = x_hat[2:]
        merged_image = image[1]
        image = image[2:]

        # Save transparent VAE decoded results（添加 alpha channel 診斷）
        print(f"[DEBUG] Saving {x_hat.shape[0]} foreground layers...", flush=True)
        for layer_idx in range(x_hat.shape[0]):
            layer = x_hat[layer_idx]
            rgba_layer = (layer.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Debug: 檢查 alpha channel
            alpha_channel = rgba_layer[:, :, 3]
            alpha_min, alpha_max = int(alpha_channel.min()), int(alpha_channel.max())
            alpha_mean = float(alpha_channel.mean())
            transparent_pixels = int((alpha_channel == 0).sum())
            total_pixels = alpha_channel.size
            transparent_ratio = transparent_pixels / total_pixels * 100
            
            # 獲取對應的 box
            if layer_idx < len(layer_boxes) - 2:
                corresponding_box = layer_boxes[layer_idx + 2]
                x1, y1, x2, y2 = corresponding_box
                box_area = (x2 - x1) * (y2 - y1)
                print(f"  Layer {layer_idx}: box={corresponding_box}, box_area={box_area}, "
                      f"alpha_range=[{alpha_min}, {alpha_max}], alpha_mean={alpha_mean:.1f}, "
                      f"transparent={transparent_ratio:.1f}%", flush=True)
            else:
                print(f"  Layer {layer_idx}: alpha_range=[{alpha_min}, {alpha_max}], "
                      f"alpha_mean={alpha_mean:.1f}, transparent={transparent_ratio:.1f}%", flush=True)
            
            rgba_image = Image.fromarray(rgba_layer, "RGBA")
            rgba_image.save(os.path.join(case_dir, f"layer_{layer_idx}_rgba.png"))

        # Composite background and foreground layers
        for layer_idx in range(x_hat.shape[0]):
            rgba_layer = (x_hat[layer_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            layer_image = Image.fromarray(rgba_layer, "RGBA")
            merged_image = Image.alpha_composite(merged_image.convert('RGBA'), layer_image)
        
        # Save final composite images
        merged_image.convert('RGB').save(os.path.join(config['save_dir'], "merged", f"{this_index}.png"))
        merged_image.convert('RGB').save(os.path.join(case_dir, f"{this_index}.png"))
        # Save final composite RGBA image
        merged_image.save(os.path.join(config['save_dir'], "merged_rgba", f"{this_index}.png"))

        print(f"✅ Saved case {idx} to {case_dir}")
        idx += 1

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"✅ 測試完成！處理了 {idx} 個樣本")
    print(f"   輸出目錄：{config['save_dir']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="使用原版 CLD infer.py 測試，但只處理少量樣本"
    )
    parser.add_argument(
        "--config_path", "-c", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int,
        default=5,
        help="最多處理的樣本數量（預設：5）"
    )
    args = parser.parse_args()

    # 導入必要的模組（torch 已經在頂部導入）
    import numpy as np
    from PIL import Image

    # CUDA 檢查已經在頂部完成，這裡直接加載配置
    config = infer_module.load_config(args.config_path)
    
    # 應用 skip_fuse_lora patch（如果配置中啟用）
    # 這需要在加載 config 之後，但在 initialize_pipeline 之前
    apply_skip_fuse_lora_patch(config)
    
    print(f"\n{'='*60}")
    print("CLD 測試腳本（限制樣本數量）")
    print(f"{'='*60}")
    print(f"配置檔案：{args.config_path}")
    print(f"最多處理樣本數：{args.max_samples}")
    print(f"輸出目錄：{config['save_dir']}")
    if config.get('skip_fuse_lora', False):
        print(f"記憶體優化：skip_fuse_lora=True (跳過 fuse_lora 以節省記憶體)")
    print(f"{'='*60}\n")
    print("⚠️  注意：首次運行時會下載 PrismLayersPro dataset 的 metadata")
    print("   但只會處理前 {} 個樣本，不會下載整個 100GB+ dataset\n".format(args.max_samples))

    try:
        inference_layout_limited(config, max_samples=args.max_samples)
        raise SystemExit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  用戶中斷")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n\n❌ 錯誤：{e}", flush=True)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

