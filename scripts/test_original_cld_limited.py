#!/usr/bin/env python3
"""
Original CLD Wrapper: Limited Sample Inference

This script acts as a wrapper around the original `infer.py` to:
1. Call the original functions dynamically.
2. Limit the number of samples at the DataLoader level.
3. Avoid downloading the massive 100GB+ dataset (streaming mode).
4. Apply memory optimizations (LoRA loading, specific GPU management).

Usage:
    cd /path/to/cld/infer
    python test_original_cld_limited.py --config_path <config.yaml> --max_samples 5
"""

import sys
import os
import time
import argparse
import importlib.util
import gc
from pathlib import Path
from collections import defaultdict
from itertools import islice, chain

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Third-party optional imports (handled in code or assumed present)
try:
    from datasets import load_dataset, concatenate_datasets, Dataset as HfDataset
    from torch.utils.data import Dataset, DataLoader
    import safetensors.torch
except ImportError:
    pass

# --- Setup Paths ---
script_path = Path(__file__).resolve()
repo_root = script_path.parent.parent
cld_root = repo_root / "third_party" / "cld"
cld_infer_dir = cld_root / "infer"
infer_py_path = cld_infer_dir / "infer.py"

# Add CLD root to sys.path to allow internal imports
sys.path.insert(0, str(cld_root))
os.chdir(str(cld_root))


# ==========================================
# 1. System & Environment Setup
# ==========================================

def apply_memory_optimization_patches():
    """
    Patches ModelMixin.from_pretrained to force bfloat16 for memory efficiency.
    This must be called BEFORE loading the infer_module to ensure all model loading uses half precision.
    """
    print("[INFO] Applying memory optimization patches (bfloat16)...", flush=True)
    try:
        from diffusers import ModelMixin

        # Store original from_pretrained method
        original_from_pretrained_func = ModelMixin.from_pretrained.__func__

        def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            """Patched from_pretrained that enforces bfloat16 for memory efficiency."""
            # Force torch_dtype=bfloat16 if not specified
            if 'torch_dtype' not in kwargs:
                kwargs['torch_dtype'] = torch.bfloat16
            elif kwargs.get('torch_dtype') != torch.bfloat16:
                # Only warn if explicitly set to something else (not None)
                if kwargs.get('torch_dtype') is not None:
                    print(f"   ⚠️  Overriding torch_dtype={kwargs['torch_dtype']} → bfloat16 for memory efficiency", flush=True)
                kwargs['torch_dtype'] = torch.bfloat16

            # Force low_cpu_mem_usage=True
            if 'low_cpu_mem_usage' not in kwargs:
                kwargs['low_cpu_mem_usage'] = True
            elif not kwargs.get('low_cpu_mem_usage'):
                print("   ⚠️  Forcing low_cpu_mem_usage=True for memory efficiency", flush=True)
                kwargs['low_cpu_mem_usage'] = True

            # Prefer safetensors if available (enables memory mapping)
            if 'use_safetensors' not in kwargs:
                kwargs['use_safetensors'] = True

            # Call original method
            return original_from_pretrained_func(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Apply monkey patch as classmethod
        ModelMixin.from_pretrained = classmethod(patched_from_pretrained)
        print("✅ Memory optimization patches applied: torch_dtype=bfloat16, low_cpu_mem_usage=True, use_safetensors=True", flush=True)
        return True

    except ImportError as e:
        print(f"⚠️  Warning: Could not apply memory optimization patches: {e}", flush=True)
        print("   Model loading may use more memory than necessary.", flush=True)
        return False
    except Exception as e:
        print(f"⚠️  Warning: Error applying memory optimization patches: {e}", flush=True)
        print("   Proceeding without patches, but memory usage may be high.", flush=True)
        return False


def setup_cuda_environment():
    """Checks CUDA availability and warns user if running on CPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {device_count}\n")
        return True, device_count
    else:
        print("\n" + "="*60)
        print("⚠️  WARNING: CUDA is not available!")
        print("="*60)
        print("CLD inference requires a GPU.")
        print("Running on CPU will be extremely slow.")
        
        response = input("Continue with CPU? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
        return False, 0


def load_modified_infer_module(infer_path, device_count):
    """
    Loads the original infer.py module but patches the hardcoded
    CUDA_VISIBLE_DEVICES setting if only 1 GPU is available.
    """
    with open(infer_path, 'r', encoding='utf-8') as f:
        infer_code = f.read()

    # Patch: If only 1 GPU, change "1" to "0" or remove the restriction
    if device_count == 1:
        if 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"' in infer_code:
            infer_code = infer_code.replace(
                'os.environ["CUDA_VISIBLE_DEVICES"] = "1"',
                'os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Modified to use GPU 0'
            )
            print("[INFO] Detected single GPU: Patching infer.py to use GPU 0 instead of 1.\n")

    spec = importlib.util.spec_from_file_location("infer_module", str(infer_path))
    module = importlib.util.module_from_spec(spec)
    exec(compile(infer_code, str(infer_path), 'exec'), module.__dict__)
    return module


# ==========================================
# 2. Monkey Patching / Optimizations
# ==========================================

def apply_lora_optimizations():
    """
    Patches CustomFluxPipeline to optimize LoRA loading:
    1. Uses safetensors directly (faster).
    2. Loads directly to GPU.
    """
    print("[DEBUG] Applying LoRA loading optimizations...", flush=True)
    try:
        from models.pipeline import CustomFluxPipeline

        if not hasattr(CustomFluxPipeline, 'lora_state_dict'):
            return

        original_lora_state_dict = CustomFluxPipeline.lora_state_dict

        @staticmethod
        def optimized_lora_state_dict(lora_path, *args, **kwargs):
            lora_path_obj = Path(lora_path)
            lora_file = None
            
            # Resolve path to safetensors
            if lora_path_obj.is_dir():
                candidates = list(lora_path_obj.glob("*.safetensors"))
                if (lora_path_obj / "pytorch_lora_weights.safetensors").exists():
                    lora_file = lora_path_obj / "pytorch_lora_weights.safetensors"
                elif candidates:
                    lora_file = candidates[0]
            elif lora_path_obj.suffix == ".safetensors":
                lora_file = lora_path_obj
            else:
                # Try finding a sibling safetensors file
                candidate = lora_path_obj.parent / f"{lora_path_obj.stem}.safetensors"
                if candidate.exists():
                    lora_file = candidate
            
            if not lora_file or not lora_file.exists():
                print(f"⚠️  Warning: Safetensors not found for {lora_path}, using fallback.")
                return original_lora_state_dict(lora_path, *args, **kwargs)

            print(f"   📦 Loading LoRA (Optimized): {lora_file}", flush=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                # Attempt direct GPU load
                try:
                    state_dict = safetensors.torch.load_file(str(lora_file), device=device)
                except TypeError:
                    # Fallback for older versions: Load CPU -> Move to GPU
                    state_dict = safetensors.torch.load_file(str(lora_file))
                    if device == "cuda":
                        for k in state_dict:
                            if isinstance(state_dict[k], torch.Tensor):
                                state_dict[k] = state_dict[k].cuda(non_blocking=True)
                        torch.cuda.synchronize()

                print(f"   ✅ LoRA loaded successfully ({len(state_dict)} keys).")
                return state_dict, None # Return tuple as expected
            except Exception as e:
                print(f"   ⚠️  Error in optimized load: {e}. Falling back.")
                return original_lora_state_dict(lora_path, *args, **kwargs)

        # Apply Patch
        CustomFluxPipeline.lora_state_dict = optimized_lora_state_dict

        # Patch load_lora_into_transformer to handle tuple return
        if hasattr(CustomFluxPipeline, 'load_lora_into_transformer'):
            original_load = CustomFluxPipeline.load_lora_into_transformer
            @staticmethod
            def wrapper_load_lora(lora_state_dict, *args, **kwargs):
                if isinstance(lora_state_dict, tuple):
                    lora_state_dict = lora_state_dict[0]
                return original_load(lora_state_dict, *args, **kwargs)
            CustomFluxPipeline.load_lora_into_transformer = wrapper_load_lora

        print("✅ LoRA I/O optimizations applied.")

    except ImportError:
        print("⚠️  Could not import pipeline for optimization.")
    except Exception as e:
        print(f"⚠️  Error applying LoRA optimizations: {e}")


def apply_gpu_fuse_optimizations():
    """
    Patches fuse_lora methods to ensure operations happen on GPU.
    """
    print("[INFO] Optimizing fuse_lora for GPU execution...", flush=True)
    try:
        from models.mmdit import CustomFluxTransformer2DModel
        from models.multiLayer_adapter import MultiLayerAdapter

        def create_optimized_fuse(original_method, class_name):
            def optimized_fuse(self, *args, **kwargs):
                print(f"[DEBUG] {class_name}.fuse_lora: Starting GPU fusion...", flush=True)
                
                # Move model to GPU if needed
                if torch.cuda.is_available():
                    self.to('cuda')
                    torch.cuda.empty_cache()
                
                start = time.time()
                result = original_method(self, *args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                print(f"  ✅ {class_name}.fuse_lora completed in {time.time() - start:.2f}s")
                return result
            return optimized_fuse

        if hasattr(CustomFluxTransformer2DModel, 'fuse_lora'):
            CustomFluxTransformer2DModel.fuse_lora = create_optimized_fuse(
                CustomFluxTransformer2DModel.fuse_lora, "Transformer"
            )
            
        if hasattr(MultiLayerAdapter, 'fuse_lora'):
            MultiLayerAdapter.fuse_lora = create_optimized_fuse(
                MultiLayerAdapter.fuse_lora, "MultiLayerAdapter"
            )

        print("✅ GPU-optimized fuse_lora patches applied.")

    except Exception as e:
        print(f"⚠️  Error optimizing fuse_lora: {e}")


def apply_skip_fuse_patch(config):
    """
    If skip_fuse_lora is True in config, disable fusing entirely.
    LoRA will function via PEFT mechanism (saves memory).
    """
    if not config.get('skip_fuse_lora', False):
        return

    print("⚠️  skip_fuse_lora=True: Disabling weight fusion to save memory.", flush=True)
    try:
        from models.mmdit import CustomFluxTransformer2DModel
        from models.multiLayer_adapter import MultiLayerAdapter

        def noop(*args, **kwargs): return None

        if hasattr(CustomFluxTransformer2DModel, 'fuse_lora'):
            CustomFluxTransformer2DModel.fuse_lora = noop
            CustomFluxTransformer2DModel.unload_lora = noop
        
        if hasattr(MultiLayerAdapter, 'fuse_lora'):
            MultiLayerAdapter.fuse_lora = noop
            MultiLayerAdapter.unload_lora = noop
            
        print("✅ fuse_lora/unload_lora disabled (PEFT mode active).")
    except Exception as e:
        print(f"⚠️  Failed to patch skip_fuse: {e}")


# ==========================================
# 3. Dataset Implementation
# ==========================================

class LimitedLayoutTrainDataset(Dataset):
    """
    Modified LayoutTrainDataset that limits sample count at initialization.
    Uses streaming to avoid downloading dataset metadata.
    """
    def __init__(self, data_dir, split="test", max_samples=None):
        print(f"[INFO] Loading PrismLayersPro dataset (split={split})...", flush=True)
        print(f"[INFO] Using streaming mode to avoid full metadata download.", flush=True)
        
        streaming_dataset = load_dataset(
            "artplus/PrismLayersPro",
            cache_dir=data_dir,
            streaming=True,
        )
        
        # Combine streams
        all_streams = [ds for _, ds in streaming_dataset.items()]
        combined_stream = chain(*all_streams)

        # Logic for selection
        if max_samples and max_samples < 100:
            print(f"[INFO] Small sample mode: taking first {max_samples} items.", flush=True)
            limited_items = list(islice(combined_stream, max_samples))
            self.dataset = HfDataset.from_list(limited_items)
        else:
            # Complex sampling logic (preserved from original)
            sample_multiplier = 10 if max_samples else 1
            target = (max_samples * sample_multiplier) if max_samples else None
            
            print(f"[INFO] Collecting samples for categorization...", flush=True)
            if target:
                items = list(islice(combined_stream, target))
            else:
                print("[WARNING] No limit set. Collecting ALL samples (slow).")
                items = list(combined_stream)
                
            full_ds = HfDataset.from_list(items)
            
            # Simple split logic based on style_category
            if "style_category" not in full_ds.column_names:
                raise ValueError("Missing 'style_category'.")

            cats = np.array(full_ds["style_category"])
            cat_indices = defaultdict(list)
            for i, c in enumerate(cats):
                cat_indices[c].append(i)

            subsets = []
            for indices in cat_indices.values():
                total = len(indices)
                p90, p95 = int(total * 0.9), int(total * 0.95)
                
                if split == "train": idxs = indices[:p90]
                elif split == "test": idxs = indices[p90:p95]
                else: idxs = indices[p95:] # val
                
                subsets.append(full_ds.select(idxs))

            combined = concatenate_datasets(subsets)
            if max_samples:
                actual = min(max_samples, len(combined))
                self.dataset = combined.select(range(actual))
            else:
                self.dataset = combined

        print(f"[INFO] Final dataset size: {len(self.dataset)}")
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Helper: Convert RGBA PIL to RGB PIL (grey background)
        def rgba2rgb(img_rgba):
            res = Image.new("RGB", img_rgba.size, (128, 128, 128))
            res.paste(img_rgba, mask=img_rgba.split()[3])
            return res

        def process_img_input(x):
            if isinstance(x, str):
                rgba = Image.open(x).convert("RGBA")
            else:
                rgba = x.convert("RGBA")
            return rgba, rgba2rgb(rgba)

        whole_rgba, whole_rgb = process_img_input(item["whole_image"])
        W, H = whole_rgba.size
        base_layout = [0, 0, W - 1, H - 1]

        layer_tensors_rgba = [self.to_tensor(whole_rgba)]
        layer_tensors_rgb = [self.to_tensor(whole_rgb)]
        layout = [base_layout]

        base_rgba, base_rgb = process_img_input(item["base_image"])
        layer_tensors_rgba.append(self.to_tensor(base_rgba))
        layer_tensors_rgb.append(self.to_tensor(base_rgb))
        layout.append(base_layout)

        for i in range(item["layer_count"]):
            key = f"layer_{i:02d}"
            l_rgba, l_rgb = process_img_input(item[key])
            w0, h0, w1, h1 = item[f"{key}_box"]

            # Create canvas
            canv_rgba = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canv_rgb = Image.new("RGB", (W, H), (128, 128, 128))

            target_w, target_h = w1 - w0, h1 - h0
            if l_rgba.size != (target_w, target_h):
                l_rgba = l_rgba.resize((target_w, target_h), Image.BILINEAR)
                l_rgb = l_rgb.resize((target_w, target_h), Image.BILINEAR)

            canv_rgba.paste(l_rgba, (w0, h0), l_rgba)
            canv_rgb.paste(l_rgb, (w0, h0))

            layer_tensors_rgba.append(self.to_tensor(canv_rgba))
            layer_tensors_rgb.append(self.to_tensor(canv_rgb))
            layout.append([w0, h0, w1, h1])

        return {
            "pixel_RGBA": layer_tensors_rgba,
            "pixel_RGB": layer_tensors_rgb,
            "whole_img": whole_rgb,
            "caption": item["whole_caption"],
            "height": H,
            "width": W,
            "layout": layout,
        }


# ==========================================
# 4. Main Inference Logic
# ==========================================

def run_inference(infer_module, config, max_samples=5):
    """
    Main inference loop with aggressive memory management.
    """
    if config.get('seed') is not None:
        infer_module.seed_everything(config['seed'])
    
    # Create directories
    save_root = Path(config['save_dir'])
    (save_root / "merged").mkdir(parents=True, exist_ok=True)
    (save_root / "merged_rgba").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load VAE ---
    print("[INFO] Loading Transparent VAE...", flush=True)
    from models.transp_vae import AutoencoderKLTransformerTraining as CustomVAE
    
    vae_args = argparse.Namespace(
        max_layers=config.get('max_layers', 48),
        decoder_arch=config.get('decoder_arch', 'vit'),
        pos_embedding=config.get('pos_embedding', 'rope'),
        layer_embedding=config.get('layer_embedding', 'rope'),
        single_layer_decoder=config.get('single_layer_decoder', None)
    )
    transp_vae = CustomVAE(vae_args)
    
    # Safe loading logic
    vae_path = config.get('transp_vae_path')
    try:
        weights = torch.load(vae_path, map_location="cpu", weights_only=False)
        if isinstance(weights, dict) and 'model' in weights:
            transp_vae.load_state_dict(weights['model'], strict=False)
    except Exception as e:
        print(f"[ERROR] Failed to load VAE: {e}")
        return

    transp_vae.eval().to(device)

    # Convert VAE to bfloat16 for memory efficiency
    if torch.cuda.is_available():
        print("[INFO] Converting VAE to bfloat16 for memory efficiency...", flush=True)
        transp_vae = transp_vae.to(torch.bfloat16)

    # --- Load Pipeline ---
    apply_skip_fuse_patch(config)
    pipeline = infer_module.initialize_pipeline(config)

    # Ensure pipeline components are also in bfloat16
    if torch.cuda.is_available():
        print("[INFO] Ensuring pipeline components use bfloat16...", flush=True)
        # Convert transformer to bfloat16 if available
        if hasattr(pipeline, 'transformer'):
            pipeline.transformer = pipeline.transformer.to(torch.bfloat16)
        # Convert VAE in pipeline to bfloat16 if available
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            pipeline.vae = pipeline.vae.to(torch.bfloat16)
    
    # --- Setup Data ---
    dataset = LimitedLayoutTrainDataset(config['data_dir'], split="test", max_samples=max_samples)
    loader = infer_module.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=infer_module.collate_fn
    )
    
    generator = torch.Generator(device=device).manual_seed(config.get('seed', 42))

    # --- Loop ---
    print(f"\n{'='*40}\nStarting Inference ({len(dataset)} samples)\n{'='*40}")
    
    for idx, batch in enumerate(loader):
        print(f"Processing case {idx}...", flush=True)
        
        # Aggressive cleanup before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        H, W = int(batch["height"][0]), int(batch["width"][0])
        adapter_img = batch["whole_img"][0]
        caption = batch["caption"][0]
        boxes = infer_module.get_input_box(batch["layout"][0])

        with torch.no_grad():
            x_hat, image_res, latents = pipeline(
                prompt=caption,
                adapter_image=adapter_img,
                adapter_conditioning_scale=0.9,
                validation_box=boxes,
                generator=generator,
                height=H, width=W,
                guidance_scale=config.get('cfg', 4.0),
                num_layers=len(boxes),
                sdxl_vae=transp_vae,
            )

        # Move to CPU immediately to free VRAM
        x_hat = (x_hat + 1) / 2
        x_hat = x_hat.squeeze(0).permute(1, 0, 2, 3).cpu().float()
        
        del latents # Free latents
        
        # Save results
        case_dir = save_root / f"case_{idx}"
        case_dir.mkdir(exist_ok=True)
        
        # 1. Whole Image & Origin
        whole_img_np = (x_hat[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(whole_img_np, "RGBA").save(case_dir / "whole_image_rgba.png")
        adapter_img.save(case_dir / "origin.png")
        
        # 2. Background
        bg_np = (x_hat[1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(bg_np, "RGBA").save(case_dir / "background_rgba.png")

        # 3. Layers
        layers_tensor = x_hat[2:]
        merged_img = image_res[1] if isinstance(image_res, list) else image_res
        if isinstance(merged_img, torch.Tensor): merged_img = merged_img.cpu()
        
        # Re-composite logic (from code) to ensure quality
        merged_pil = Image.fromarray(bg_np, "RGBA") # Start with background

        for l_idx in range(layers_tensor.shape[0]):
            l_np = (layers_tensor[l_idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            l_pil = Image.fromarray(l_np, "RGBA")
            l_pil.save(case_dir / f"layer_{l_idx}_rgba.png")
            merged_pil = Image.alpha_composite(merged_pil, l_pil)

        merged_pil.convert('RGB').save(save_root / "merged" / f"case_{idx}.png")
        merged_pil.convert('RGB').save(case_dir / f"case_{idx}.png")
        merged_pil.save(save_root / "merged_rgba" / f"case_{idx}.png")

        print(f"✅ Saved case {idx}")
        
        # Cleanup
        del x_hat, image_res, layers_tensor, merged_pil
        
    print("\n✅ Inference Complete.")


# ==========================================
# 5. Entry Point
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLD Limited Inference Wrapper")
    parser.add_argument("--config_path", "-c", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--max_samples", "-n", type=int, default=5, help="Max samples to process")
    args = parser.parse_args()

    # 1. Setup Environment
    cuda_ok, dev_count = setup_cuda_environment()

    # 2. Apply Memory Optimization Patches (MUST be before loading infer_module)
    # This patches ModelMixin.from_pretrained to force bfloat16
    apply_memory_optimization_patches()

    # 3. Load Original Module
    infer_module = load_modified_infer_module(infer_py_path, dev_count)
    
    # 3. Apply Optimizations
    apply_lora_optimizations()
    apply_gpu_fuse_optimizations()
    
    # 4. Load Config & Run
    config = infer_module.load_config(args.config_path)
    
    # Log memory settings
    if config.get('skip_fuse_lora'):
        print("[CONFIG] skip_fuse_lora=True (Memory Saving Mode)")

    try:
        run_inference(infer_module, config, args.max_samples)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)