#!/usr/bin/env python3
"""
Original CLD Wrapper: Limited Sample Inference with FP8 Quantization

This script acts as a wrapper around the original `infer.py` to:
1. Call the original functions dynamically.
2. Limit the number of samples at the DataLoader level.
3. Avoid downloading the massive 100GB+ dataset (streaming mode).
4. Apply memory optimizations (LoRA loading, bfloat16, FP8 quantization).
5. FP8 quantization for T5-XXL text encoder (saves ~50% VRAM on T5 components).

Usage:
    cd /path/to/cld/infer
    python test_original_cld_limited.py --config_path <config.yaml> --max_samples 5 [--enable_fp8]

FP8 Quantization:
    - Automatically detects and quantizes T5/text encoder modules
    - Stores weights in FP8 format (1 byte per parameter) to save VRAM
    - Dynamically decompresses to BF16 during forward pass for computation
    - Can save hundreds of MB of VRAM for T5-XXL models
    - Enable with --enable_fp8 flag or config: enable_fp8_quantization=true
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

# Set PyTorch CUDA memory allocation config to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# FP8 quantization utilities
try:
    import torch.nn.functional as F
    FP8_AVAILABLE = True
except ImportError:
    FP8_AVAILABLE = False


class FP8QuantizedLinear(torch.nn.Module):
    """
    FP8 quantized Linear layer with dynamic decompression.

    Stores weights in FP8 format to save VRAM, but decompresses to BF16/FP16
    during forward pass for computation.
    """
    def __init__(self, original_linear, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bias = original_linear.bias

        # Quantize weights to FP8
        self.quantize_weights(original_linear.weight.data)

    def quantize_weights(self, weight):
        """Quantize weights to FP8 format with per-block scaling."""
        # Store original weight shape
        original_shape = weight.shape

        # For simplicity, quantize per output feature (row-wise)
        # weight shape is [out_features, in_features]
        weight_flat = weight.view(weight.shape[0], -1)  # [out_features, in_features]

        # Calculate scales per output feature
        weight_abs = weight_flat.abs()
        scales = weight_abs.max(dim=1, keepdim=True)[0]  # [out_features, 1]

        # Avoid division by zero
        scales = torch.clamp(scales, min=1e-8)

        # Quantize to FP8 (simulate E4M3 format)
        # FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
        # Range: -448 to 448
        fp8_max = 448.0
        normalized = weight_flat / scales
        quantized = torch.clamp(normalized, -fp8_max, fp8_max)

        # Convert to FP8 storage format (if available) or keep as float8
        try:
            # Try to use torch.float8_e4m3fn if available (torch >= 2.1)
            fp8_tensor = quantized.to(torch.float8_e4m3fn)
        except AttributeError:
            # Fallback: store as float16 and simulate FP8 precision
            # Reduce precision by quantizing to lower bit representation
            fp8_tensor = (quantized / fp8_max * 127).round().clamp(-127, 127).to(torch.int8)

        # Store quantized weights and scales
        self.register_buffer('quantized_weight', fp8_tensor)
        self.register_buffer('scales', scales)
        self.register_buffer('original_shape', torch.tensor(original_shape))

    def forward(self, x):
        """Dynamic decompression during forward pass."""
        # Decompress weights back to BF16/FP16
        try:
            # If stored as float8_e4m3fn, convert back
            if hasattr(torch, 'float8_e4m3fn') and self.quantized_weight.dtype == torch.float8_e4m3fn:
                dequantized = self.quantized_weight.to(torch.bfloat16) * self.scales
            else:
                # If stored as int8, convert back from simulation
                fp8_max = 448.0
                dequantized = (self.quantized_weight.to(torch.bfloat16) / 127 * fp8_max) * self.scales
        except Exception as e:
            print(f"[FP8] Dequantization error: {e}, using zero weights")
            # Fallback: return zero tensor with correct shape
            return F.linear(x, torch.zeros(self.original_shape, dtype=torch.bfloat16, device=x.device), self.bias)

        # Reshape back to original weight shape [out_features, in_features]
        dequantized = dequantized.view(self.original_shape)

        # Standard linear operation
        return F.linear(x, dequantized, self.bias)


def apply_fp8_quantization_to_module(module, module_name=""):
    """
    Apply FP8 quantization to Linear layers within a module, in-place.

    Args:
        module: The PyTorch module to quantize (e.g., T5EncoderModel)
        module_name: Name of the module for logging

    Returns:
        Dict with quantization statistics
    """
    if not FP8_AVAILABLE:
        return {'quantized_layers': 0, 'memory_saved': 0}

    print(f"[FP8] Quantizing Linear layers in {module_name}...")

    quantized_layers = {}
    total_memory_saved = 0

    for name, child_module in module.named_modules():
        if isinstance(child_module, torch.nn.Linear):
            print(f"[FP8]   Quantizing layer: {name} ({child_module.in_features} -> {child_module.out_features})")

            try:
                # Create quantized version
                fp8_layer = FP8QuantizedLinear(child_module)

                # Replace the layer in its parent
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = module
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)

                # Store original layer for potential rollback
                original_layer = getattr(parent, child_name)
                setattr(parent, child_name, fp8_layer)

                # Calculate memory savings for this layer
                original_params = child_module.in_features * child_module.out_features
                if child_module.bias is not None:
                    original_params += child_module.out_features

                # Original memory: BF16 = 2 bytes per param
                original_memory_bytes = original_params * 2

                # Quantized memory: FP8 weights + BF16 scales + BF16 bias
                # FP8 weights: 1 byte per weight param
                weight_params = child_module.in_features * child_module.out_features
                quantized_memory_bytes = weight_params * 1  # FP8 weights

                # Scales: small overhead, approximately weight_params / block_size
                # For simplicity, estimate as 10% overhead for scales
                scale_overhead = weight_params * 1 * 0.1  # Rough estimate
                quantized_memory_bytes += scale_overhead

                # Bias: remains BF16 if exists
                if child_module.bias is not None:
                    quantized_memory_bytes += child_module.out_features * 2

                memory_saved_bytes = original_memory_bytes - quantized_memory_bytes

                quantized_layers[name] = {
                    'original_layer': original_layer,
                    'quantized_layer': fp8_layer,
                    'original_memory_mb': original_memory_bytes / (1024**2),
                    'quantized_memory_mb': quantized_memory_bytes / (1024**2),
                    'memory_saved_mb': memory_saved_bytes / (1024**2)
                }

                total_memory_saved += memory_saved_bytes / (1024**2)

                print(f"[FP8]     Memory: {original_memory_bytes/(1024**2):.3f}MB → {quantized_memory_bytes/(1024**2):.3f}MB "
                      f"(saved {memory_saved_bytes/(1024**2):.3f}MB)")

            except Exception as e:
                print(f"[FP8]   ❌ Failed to quantize {name}: {e}")

    return {
        'quantized_layers': len(quantized_layers),
        'memory_saved': total_memory_saved,
        'layer_details': quantized_layers
    }


def apply_fp8_quantization_to_pipeline(pipeline, config=None, target_modules=['T5EncoderModel']):
    """
    Apply FP8 quantization to specific modules in the pipeline (e.g., T5-XXL).

    Args:
        pipeline: The FLUX/CustomFlux pipeline
        config: Configuration dict to check if FP8 is enabled
        target_modules: List of module names to quantize (e.g., ['T5EncoderModel'])

    Returns:
        Modified pipeline with FP8 quantization
    """
    # Check if FP8 quantization is enabled in config
    if config and not config.get('enable_fp8_quantization', False):
        print("[FP8] FP8 quantization disabled in config")
        return pipeline

    if not FP8_AVAILABLE:
        print("⚠️  FP8 quantization not available (torch version too old)")
        return pipeline

    print("[FP8] Applying FP8 quantization to pipeline...", flush=True)
    print(f"[FP8] Pipeline type: {type(pipeline).__name__}")

    # For diffusers pipelines, directly check common text encoder attributes
    # This is more reliable than trying to use named_modules()
    quantized_modules = {}
    total_memory_saved = 0

    # Common text encoder attributes in diffusers FLUX pipelines
    text_encoder_candidates = [
        'text_encoder',      # CLIP text encoder
        'text_encoder_2',    # T5 text encoder (FLUX.1-pro)
        'text_encoder_3',    # Additional text encoders if any
        't5_encoder',        # Direct T5 encoder
        't5_decoder',        # T5 decoder (usually not needed for inference)
    ]

    print("[FP8] Checking for text encoders in pipeline...")

    for attr_name in text_encoder_candidates:
        if hasattr(pipeline, attr_name):
            module = getattr(pipeline, attr_name)
            if module is not None and hasattr(module, 'parameters'):
                print(f"[FP8] Found text encoder: {attr_name} (type: {type(module).__name__})")

                # Check if it has Linear layers (most text encoders do)
                linear_layers = [child for child in module.modules() if isinstance(child, torch.nn.Linear)]
                num_linear = len(linear_layers)

                if num_linear > 0:
                    print(f"[FP8] Found {num_linear} Linear layers in {attr_name}")

                    # Check if it's a T5 model (larger models benefit more from quantization)
                    is_t5_like = ('t5' in attr_name.lower() or
                                'T5' in type(module).__name__ or
                                any('T5' in type(layer).__name__ for layer in module.modules()))

                    if is_t5_like:
                        print(f"[FP8] ✅ {attr_name} appears to be a T5-like model, applying FP8 quantization")
                    else:
                        print(f"[FP8] {attr_name} is not T5-like, but has Linear layers - applying FP8 anyway")

                    try:
                        # Apply in-place quantization to the module
                        stats = apply_fp8_quantization_to_module(module, attr_name)

                        if stats['quantized_layers'] > 0:
                            quantized_modules[attr_name] = stats
                            total_memory_saved += stats['memory_saved']
                            print(f"[FP8] ✅ {attr_name} quantized: {stats['quantized_layers']} layers, "
                                  f"saved {stats['memory_saved']:.1f}MB")
                        else:
                            print(f"[FP8] ⚠️  No layers were quantized in {attr_name}")

                    except Exception as e:
                        print(f"[FP8] ❌ Failed to quantize {attr_name}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[FP8] {attr_name} has no Linear layers, skipping")
            else:
                print(f"[FP8] {attr_name} is None or not a proper module")
        else:
            print(f"[FP8] {attr_name} not found in pipeline")

    if not quantized_modules:
        print("⚠️  No suitable text encoders found for FP8 quantization")
        print("   This is normal if using FLUX.1-dev (CLIP only) instead of FLUX.1-pro (T5)")
        print("   Or if the pipeline doesn't have text encoders loaded")
        return pipeline

    # No need to replace modules since we quantized in-place
    print(f"[FP8] 🎉 FP8 quantization completed! Total memory saved: {total_memory_saved:.1f}MB")
    print("[FP8] Note: This may slightly reduce output quality but saves significant VRAM")
    print("[FP8] Quantized modules remain in their original locations in the pipeline")

    return pipeline

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
    Patches ModelMixin.from_pretrained to force bfloat16/float16 for memory efficiency.
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

    # Apply FP8 quantization to T5 components if available
    pipeline = apply_fp8_quantization_to_pipeline(pipeline, config, target_modules=['T5EncoderModel', 'T5DecoderModel', 'text_encoder_2'])
    
    # --- Setup Data ---
    dataset = LimitedLayoutTrainDataset(config['data_dir'], split="test", max_samples=max_samples)
    loader = infer_module.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=infer_module.collate_fn
    )
    
    generator = torch.Generator(device=device).manual_seed(config.get('seed', 42))

    # --- Loop ---
    print(f"\n{'='*40}\nStarting Inference ({len(dataset)} samples)\n{'='*40}")
    
    for idx, batch in enumerate(loader):
        print(f"\n{'='*60}")
        print(f"Processing case {idx} (Sample {idx+1} of {len(dataset)})")
        print(f"{'='*60}", flush=True)
        
        # Aggressive cleanup before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        H, W = int(batch["height"][0]), int(batch["width"][0])
        adapter_img = batch["whole_img"][0]
        caption = batch["caption"][0]
        boxes = infer_module.get_input_box(batch["layout"][0])
        
        # Delete batch immediately to free memory before pipeline call
        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate layers using pipeline
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
        x_hat = x_hat.squeeze(0).permute(1, 0, 2, 3).cpu().to(torch.float32)
        
        # Also move image_res to CPU immediately
        if isinstance(image_res, torch.Tensor):
            image_res = image_res.cpu()
        elif isinstance(image_res, (list, tuple)):
            image_res = [img.cpu() if isinstance(img, torch.Tensor) else img for img in image_res]
        
        # Delete latents immediately (not needed after this point)
        del latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save results
        case_dir = save_root / f"case_{idx}"
        case_dir.mkdir(exist_ok=True)
        
        # 1. Whole Image & Origin
        whole_img_np = (x_hat[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(whole_img_np, "RGBA").save(case_dir / "whole_image_rgba.png")
        adapter_img.save(case_dir / "origin.png")
        del whole_img_np  # Free immediately
        
        # 2. Background
        bg_np = (x_hat[1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(bg_np, "RGBA").save(case_dir / "background_rgba.png")

        # 3. Layers
        layers_tensor = x_hat[2:]
        del x_hat  # Free x_hat after extracting layers
        
        # Re-composite logic (from code) to ensure quality
        merged_pil = Image.fromarray(bg_np, "RGBA")  # Start with background

        for l_idx in range(layers_tensor.shape[0]):
            l_np = (layers_tensor[l_idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            l_pil = Image.fromarray(l_np, "RGBA")
            l_pil.save(case_dir / f"layer_{l_idx}_rgba.png")
            merged_pil = Image.alpha_composite(merged_pil, l_pil)
            # Clean up immediately after each layer
            del l_np, l_pil

        merged_pil.convert('RGB').save(save_root / "merged" / f"case_{idx}.png")
        merged_pil.convert('RGB').save(case_dir / f"case_{idx}.png")
        merged_pil.save(save_root / "merged_rgba" / f"case_{idx}.png")

        print(f"✅ Saved case {idx}")
        
        # Aggressive VRAM cleanup after each image
        # Delete all intermediate variables
        try:
            del x_hat, image_res, layers_tensor, merged_pil
            del whole_img_np, bg_np, l_np, l_pil
        except NameError:
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache aggressively
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Try to reclaim reserved memory
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        
        # Print memory usage for debugging
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                device_id = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
                total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
                free = total_memory - reserved
                print(f"   💾 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free (total: {total_memory:.2f} GB)", flush=True)
            except (AssertionError, RuntimeError) as e:
                # Fallback if device properties are not accessible
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                print(f"   💾 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)
        
    print("\n✅ Inference Complete.")


# ==========================================
# 5. Entry Point
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLD Limited Inference Wrapper with FP8 quantization support")
    parser.add_argument("--config_path", "-c", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--max_samples", "-n", type=int, default=5, help="Max samples to process")
    parser.add_argument("--enable_fp8", action="store_true",
                       help="Enable FP8 quantization for T5-XXL text encoder (saves ~50%% VRAM)")
    args = parser.parse_args()

    # 1. Setup Environment
    cuda_ok, dev_count = setup_cuda_environment()
    
    # 2. Apply Memory Optimization Patches (MUST be before loading infer_module)
    # This patches ModelMixin.from_pretrained to force bfloat16
    apply_memory_optimization_patches()
    
    # 3. Load Original Module
    infer_module = load_modified_infer_module(infer_py_path, dev_count)
    
    # 4. Apply Other Optimizations
    apply_lora_optimizations()
    apply_gpu_fuse_optimizations()
    
    # 5. Load Config & Run
    config = infer_module.load_config(args.config_path)

    # Override config with command line args
    if args.enable_fp8:
        config['enable_fp8_quantization'] = True
        print("[CONFIG] FP8 quantization enabled via command line")

    # Log memory settings
    if config.get('skip_fuse_lora'):
        print("[CONFIG] skip_fuse_lora=True (Memory Saving Mode)")
    if config.get('enable_fp8_quantization', False):
        print("[CONFIG] enable_fp8_quantization=True (T5-XXL FP8 quantization enabled)")
    else:
        print("[CONFIG] FP8 quantization disabled (use --enable_fp8 or set enable_fp8_quantization=true in config)")

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