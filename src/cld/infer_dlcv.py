from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path
from types import ModuleType


def _find_repo_root(start: Path) -> Path:
    """
    Find repo root from the current file location.
    Condition: contains third_party/cld/infer/infer.py.
    """
    for p in [start, *start.parents]:
        if (p / "third_party" / "cld" / "infer" / "infer.py").exists():
            return p
    # fallback: assume src/cld_generation/infer_dlcv.py -> repo_root is parents[2]
    return start.parents[2]


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    """
    Load Python module from file path, avoid the problem that third_party/cld doesn't have package __init__.py.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project-level CLD inference runner.\n"
            "This script is placed in src/cld_generation, is a wrapper: reuse third_party/cld/infer/infer.py as much as possible, "
            "avoid copying large model initialization and inference logic."
        )
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default=None,
        help="YAML config path (e.g. configs/exp001/cld/infer.yaml).",
    )
    parser.add_argument(
        "--cld_infer_py",
        type=str,
        default=None,
        help="third_party CLD's infer.py path (default: <repo_root>/third_party/cld/infer/infer.py).",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start processing from the Nth JSON file (0-indexed). Useful for resuming interrupted runs. (default: 0)",
    )
    args = parser.parse_args()

    # Don't hardcode CUDA_VISIBLE_DEVICES here, leave it to scheduler/launcher
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Check CUDA availability before proceeding
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        
        if not cuda_available:
            print("❌ CUDA is not available in this environment")
            return 1
        
        if cuda_device_count == 0:
            print("❌ CUDA is available but no devices are visible")
            return 1
        
        print(f"✅ CUDA available: {cuda_device_count} device(s)")
    except ImportError:
        print("⚠️  Warning: torch not found. CUDA availability cannot be checked.")
        print("   Proceeding anyway, but CLD inference will likely fail without GPU.")

    this_file = Path(__file__).resolve()
    repo_root = _find_repo_root(this_file.parent)

    # Default config
    default_cfg = repo_root / "configs" / "exp001" / "cld" / "infer.yaml"
    config_path = Path(args.config_path).expanduser() if args.config_path else default_cfg
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    # Load third_party cld infer.py
    cld_infer_py = Path(args.cld_infer_py).expanduser() if args.cld_infer_py else (repo_root / "third_party" / "cld" / "infer" / "infer.py")
    if not cld_infer_py.is_absolute():
        cld_infer_py = (repo_root / cld_infer_py).resolve()
    if not cld_infer_py.exists():
        raise FileNotFoundError(f"CLD infer.py not found: {cld_infer_py}")

    cld_root = cld_infer_py.parent.parent  # .../third_party/cld
    if str(cld_root) not in sys.path:
        sys.path.insert(0, str(cld_root))

    # Memory optimization: Monkey patch from_pretrained BEFORE loading cld_infer module
    # This ensures that when infer.py calls from_pretrained, it will use our optimized version
    # This prevents memory doubling (24GB -> 48GB) and enables safetensors memory mapping
    try:
        import torch
        from diffusers import ModelMixin
        
        # Store original from_pretrained method (it's already a classmethod)
        # We need to get the underlying function to properly wrap it
        original_modelmixin_from_pretrained_func = ModelMixin.from_pretrained.__func__
        
        def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            """Patched from_pretrained that enforces memory optimizations."""
            # Force torch_dtype=bfloat16 if not specified
            if 'torch_dtype' not in kwargs:
                kwargs['torch_dtype'] = torch.bfloat16
            elif kwargs.get('torch_dtype') != torch.bfloat16:
                print(f"⚠️  Warning: torch_dtype is {kwargs['torch_dtype']}, forcing bfloat16 for memory efficiency")
                kwargs['torch_dtype'] = torch.bfloat16
            
            # Force low_cpu_mem_usage=True
            if 'low_cpu_mem_usage' not in kwargs:
                kwargs['low_cpu_mem_usage'] = True
            elif not kwargs.get('low_cpu_mem_usage'):
                print("⚠️  Warning: low_cpu_mem_usage=False, forcing True for memory efficiency")
                kwargs['low_cpu_mem_usage'] = True
            
            # Prefer safetensors if available (enables memory mapping)
            if 'use_safetensors' not in kwargs:
                kwargs['use_safetensors'] = True
            
            # Call original method with correct signature
            return original_modelmixin_from_pretrained_func(cls, pretrained_model_name_or_path, *args, **kwargs)
        
        # Apply monkey patch as classmethod
        ModelMixin.from_pretrained = classmethod(patched_from_pretrained)
        print("✅ Applied memory optimization patches: torch_dtype=bfloat16, low_cpu_mem_usage=True, use_safetensors=True")
        
    except ImportError as e:
        print(f"⚠️  Warning: Could not apply memory optimization patches: {e}")
        print("   Model loading may use more memory than necessary.")
    except Exception as e:
        print(f"⚠️  Warning: Error applying memory optimization patches: {e}")
        print("   Proceeding without patches, but memory usage may be high.")

    # Important: CLD infer.py uses `from models...` / `from tools...`, need to set cwd / sys.path to cld_root
    # Here we only change cwd to cld_root (same as finals/CLD/infer/infer_dlcv.py), ensure consistent relative path/resource reading.
    os.chdir(str(cld_root))

    cld_infer = _load_module_from_path("cld_infer", cld_infer_py)

    # Optimize LoRA loading: Monkey patch CustomFluxPipeline.lora_state_dict AFTER loading cld_infer module
    # This ensures LoRA weights are loaded directly to GPU using safetensors for faster loading
    print("[DEBUG] Starting LoRA optimization setup...", flush=True)
    CustomFluxPipeline = None  # Will be set in try block
    try:
        import torch
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
            
    except ImportError as e:
        print(f"⚠️  Warning: Could not apply LoRA loading optimization: {e}")
        print("   LoRA loading will use default method (may be slower)")
    except Exception as e:
        print(f"⚠️  Warning: Error applying LoRA loading optimization: {e}")
        print("   LoRA loading will use default method (may be slower)")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Final CUDA check and fix before calling inference_layout
    # This ensures CUDA is properly initialized in the current process
    try:
        import torch
        
        if torch.cuda.is_available() and torch.cuda.device_count() == 0:
            print("\n⚠️  Warning: CUDA is available but no devices are visible.")
            
            # Try to initialize CUDA by creating a tensor
            try:
                _ = torch.zeros(1).cuda()
                print("   ✅ CUDA context initialized successfully")
            except RuntimeError as e:
                print(f"   ❌ Failed to initialize CUDA context: {e}")
                return 1
            
            # Monkey patch torch.load to handle device_count() == 0 case AND memory optimization
            # This fixes the issue where third_party/cld/infer/infer.py uses
            # torch.load(..., map_location=torch.device("cuda")) when device_count() == 0
            # We'll load on CPU first, then move to CUDA if device becomes available
            # Also ensures weights_only=True for security and memory efficiency
            original_torch_load = torch.load
            
            def patched_torch_load(*args, **kwargs):
                """Patched torch.load that handles CUDA device issues, weights_only errors, and memory optimization."""
                # Handle map_location for CUDA device issues first
                if "map_location" in kwargs:
                    map_loc = kwargs["map_location"]
                    if isinstance(map_loc, torch.device) and map_loc.type == "cuda":
                        if torch.cuda.device_count() == 0:
                            # Load on CPU first
                            kwargs["map_location"] = "cpu"
                            result = original_torch_load(*args, **kwargs)
                            # Try to move to CUDA if device becomes available
                            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                                try:
                                    if isinstance(result, dict):
                                        return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
                                    elif isinstance(result, torch.Tensor):
                                        return result.cuda()
                                except Exception:
                                    # If CUDA still not available, return CPU version
                                    pass
                            return result
                
                # Handle weights_only error (CLD checkpoints contain argparse.Namespace)
                # Try with weights_only=False if weights_only=True fails
                weights_only_set = "weights_only" in kwargs
                if not weights_only_set:
                    # Default: try weights_only=True first for security
                    kwargs["weights_only"] = True
                
                try:
                    return original_torch_load(*args, **kwargs)
                except (pickle.UnpicklingError, RuntimeError) as e:
                    error_msg = str(e)
                    # Check if it's a weights_only error
                    if "weights_only" in error_msg or "Unsupported global" in error_msg or "argparse.Namespace" in error_msg:
                        # Retry with weights_only=False (CLD checkpoints contain non-standard objects)
                        kwargs["weights_only"] = False
                        return original_torch_load(*args, **kwargs)
                    # Re-raise if it's a different error
                    raise
            
            # Apply monkey patch
            torch.load = patched_torch_load
            print("   🔧 Applied monkey patch for torch.load to handle CUDA device issues and memory optimization")
            
    except ImportError:
        pass  # torch check already done above

    # Load config
    config = cld_infer.load_config(str(config_path))
    
    # Import necessary functions from cld_infer module
    original_initialize_pipeline = cld_infer.initialize_pipeline
    
    # Override get_input_box to use 16px quantization
    # This ensures bbox coordinates and dimensions are multiples of 16px
    def get_input_box(layer_boxes, img_width=None, img_height=None):
        """
        Quantize xyxy boxes to 16px grid.
        
        This ensures bbox coordinates and dimensions are multiples of 16px,
        which is required for proper CLD inference processing.
        
        IMPORTANT: This function ensures that quantized boxes have minimum size of 16x16 pixels.
        
        Args:
            layer_boxes: List of [x1, y1, x2, y2] boxes
            img_width: Image width (optional, for boundary clamping)
            img_height: Image height (optional, for boundary clamping)
        
        Returns:
            List of quantized boxes as (x1, y1, x2, y2) tuples
        """
        list_layer_box = []
        for layer_box in layer_boxes:
            min_row, max_row = layer_box[1], layer_box[3]
            min_col, max_col = layer_box[0], layer_box[2]
            
            # Floor to nearest multiple of 16 (min coordinates)
            quantized_min_row = (int(min_row) // 16) * 16
            quantized_min_col = (int(min_col) // 16) * 16
            
            # Ceil to nearest multiple of 16 (max coordinates)
            # Use ((max + 15) // 16) * 16 instead of ((max // 16) + 1) * 16
            # This preserves already-aligned coordinates (e.g., 800 stays 800, not 816)
            quantized_max_row = ((int(max_row) + 15) // 16) * 16
            quantized_max_col = ((int(max_col) + 15) // 16) * 16
            
            # Ensure minimum size of 16x16
            # If width or height is 0 after quantization, expand by 16 pixels
            if quantized_max_col <= quantized_min_col:
                quantized_max_col = quantized_min_col + 16
            if quantized_max_row <= quantized_min_row:
                quantized_max_row = quantized_min_row + 16
            
            # Clamp to image boundaries if provided
            if img_width is not None:
                quantized_min_col = max(0, min(quantized_min_col, img_width - 16))
                quantized_max_col = max(quantized_min_col + 16, min(quantized_max_col, img_width))
            if img_height is not None:
                quantized_min_row = max(0, min(quantized_min_row, img_height - 16))
                quantized_max_row = max(quantized_min_row + 16, min(quantized_max_row, img_height))
            
            list_layer_box.append((quantized_min_col, quantized_min_row, quantized_max_col, quantized_max_row))
        return list_layer_box
    
    # Wrap initialize_pipeline with timing and debug info
    # Also monkey patch key operations to track timing
    import time
    
    # Monkey patch load_lora_into_transformer, fuse_lora, unload_lora to add timing
    # Only if CustomFluxPipeline was successfully imported
    if CustomFluxPipeline is not None and hasattr(CustomFluxPipeline, 'load_lora_into_transformer'):
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
    
    # Patch fuse_lora and unload_lora on transformer objects
    # We'll patch them on the CustomFluxTransformer2DModel class
    try:
        from models.mmdit import CustomFluxTransformer2DModel
        
        if hasattr(CustomFluxTransformer2DModel, 'fuse_lora'):
            original_fuse_lora = CustomFluxTransformer2DModel.fuse_lora
            
            def timed_fuse_lora(self, *args, **kwargs):
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
                print("[DEBUG] unload_lora: Starting...", flush=True)
                start = time.time()
                result = original_unload_lora(self, *args, **kwargs)
                elapsed = time.time() - start
                print(f"[DEBUG] unload_lora: Completed in {elapsed:.2f}s", flush=True)
                return result
            
            CustomFluxTransformer2DModel.unload_lora = timed_unload_lora
    except ImportError:
        print("[DEBUG] Could not patch fuse_lora/unload_lora (models.mmdit not available)", flush=True)
    
    # Load config
    config = cld_infer.load_config(str(config_path))
    
    # Store args for use in inference_layout_pipeline closure
    _args_start_from = args.start_from
    
    def initialize_pipeline_with_timing(config):
        """Wrapped initialize_pipeline with detailed timing information."""
        print("[DEBUG] initialize_pipeline: Starting...", flush=True)
        total_start = time.time()
        
        print("[DEBUG] initialize_pipeline: Calling original function...", flush=True)
        result = original_initialize_pipeline(config)
        total_elapsed = time.time() - total_start
        print(f"[DEBUG] initialize_pipeline: Completed in {total_elapsed:.2f}s total", flush=True)
        return result
    
    initialize_pipeline = initialize_pipeline_with_timing
    
    # Import seed_everything from tools.tools (it's imported in infer.py but not exposed as attribute)
    from tools.tools import seed_everything
    
    # Import PipelineDataset - we always use it to avoid downloading HuggingFace datasets
    repo_root_src = repo_root / "src"
    if str(repo_root_src) not in sys.path:
        sys.path.insert(0, str(repo_root_src))
    
    from data.custom_cld_dataset import PipelineDataset, collate_fn_pipeline
    
    # Import other dependencies
    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader
    # Implement our own inference_layout that uses PipelineDataset
    # This avoids calling cld_infer.inference_layout which would use LayoutTrainDataset
    # and download the entire HuggingFace dataset
    @torch.no_grad()
    def inference_layout_pipeline(config):
        """Custom inference_layout that uses PipelineDataset instead of LayoutTrainDataset."""
        if config.get('seed') is not None:
            seed_everything(config['seed'])
        
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['save_dir'], "merged"), exist_ok=True)
        os.makedirs(os.path.join(config['save_dir'], "merged_rgba"), exist_ok=True)

        # Load transparent VAE
        print("[INFO] Loading Transparent VAE...", flush=True)
        
        import argparse
        from models.transp_vae import AutoencoderKLTransformerTraining as CustomVAE
        
        vae_args = argparse.Namespace(
            max_layers=config.get('max_layers', 48),
            decoder_arch=config.get('decoder_arch', 'vit'),
            pos_embedding=config.get('pos_embedding', 'rope'),
            layer_embedding=config.get('layer_embedding', 'rope'),
            single_layer_decoder=config.get('single_layer_decoder', None)
        )
        transp_vae = CustomVAE(vae_args)
        transp_vae_path = config.get('transp_vae_path')
        transp_vae_weights = torch.load(transp_vae_path, map_location=torch.device("cuda"))
        missing_keys, unexpected_keys = transp_vae.load_state_dict(transp_vae_weights['model'], strict=False)
        if missing_keys or unexpected_keys:
            print(f"[WARNING] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        transp_vae.eval()
        transp_vae = transp_vae.to(torch.device("cuda"))
        print("[INFO] Transparent VAE loaded successfully.", flush=True)

        print("[DEBUG] About to call initialize_pipeline (LoRA optimization should be active)...", flush=True)
        pipeline = initialize_pipeline(config)
        print("[DEBUG] initialize_pipeline completed", flush=True)
        
        # Note: Flash Attention 2 is not compatible with CLD's custom transformer
        # CLD's transformer requires:
        # 1. image_rotary_emb parameter in cross_attention_kwargs
        # 2. Attention to return two values (attn_output, context_attn_output)
        # AttnProcessor2_0 does not support these requirements
        # Therefore, we skip Flash Attention 2 and use the default attention processor
        print("[INFO] Using default attention processor (Flash Attention 2 not compatible with CLD transformer)", flush=True)
        
        # Enable Tiled VAE decoding if configured
        enable_tiled_vae = config.get('enable_tiled_vae', False)
        vae_tile_size = config.get('vae_tile_size', 512)  # Default tile size

        if enable_tiled_vae:
            print(f"[INFO] Tiled VAE decoding enabled (tile_size={vae_tile_size})", flush=True)
            # Try to use the built-in enable_tiling method first
            if hasattr(pipeline.vae, 'enable_tiling'):
                try:
                    pipeline.vae.enable_tiling()
                    print(f"[INFO] Used built-in VAE tiling with tile size {vae_tile_size}", flush=True)
                except Exception as e:
                    print(f"[WARNING] Built-in VAE tiling failed: {e}, falling back to custom implementation", flush=True)
                    enable_tiled_vae = True  # Keep flag for custom implementation
            else:
                print("[INFO] VAE does not have built-in tiling, using custom implementation", flush=True)

        # Monkey patch pipeline's VAE decode for custom tiled decoding if needed
        # Note: Pipeline already splits latents into segments (line 791 in pipeline.py)
        # We'll optimize the segment processing to use tiled decoding
        if enable_tiled_vae and hasattr(pipeline, 'vae') and not hasattr(pipeline.vae, 'enable_tiling'):
            # Store original decode method
            original_vae_decode = pipeline.vae.decode
            
            def optimized_vae_decode_wrapper(latents, return_dict=True):
                """
                Optimized VAE decode wrapper that supports:
                Tiled decoding: decode large images in tiles to reduce VRAM
                
                Note: Results are kept on GPU because pipeline needs GPU tensors for subsequent processing.
                CPU offloading happens after pipeline returns, in the main inference loop.
                """
                # Check if latents have multiple layers (shape: [B*num_layers, C, H, W])
                if len(latents.shape) == 4:
                    B, C, H, W = latents.shape
                    
                    # Check if tiled decoding should be used
                    # If enable_tiled_vae is True, always use tiled decoding (regardless of image size)
                    # This helps reduce VRAM usage even for smaller images
                    use_tiled = enable_tiled_vae
                    
                    if use_tiled:
                        # Use tiled decoding for large images
                        # Decode each layer with tiled approach
                        layer_decoded = []
                        for layer_idx in range(B):
                            layer_latent = latents[layer_idx:layer_idx+1]
                            # Use tiled approach for each layer
                            B_layer, C_layer, H_layer, W_layer = layer_latent.shape
                            num_tiles_h = (H_layer + vae_tile_size - 1) // vae_tile_size
                            num_tiles_w = (W_layer + vae_tile_size - 1) // vae_tile_size
                            overlap = 64
                            
                            # Define tiled decode helper inline to access pipeline.vae
                            def decode_tile(tile_latents):
                                """Helper to decode a single tile"""
                                return original_vae_decode(tile_latents, return_dict=False)[0]
                            
                            decoded_tiles = []
                            for i in range(num_tiles_h):
                                row_tiles = []
                                for j in range(num_tiles_w):
                                    h_start = max(0, i * vae_tile_size - overlap)
                                    h_end = min(H_layer, (i + 1) * vae_tile_size + overlap)
                                    w_start = max(0, j * vae_tile_size - overlap)
                                    w_end = min(W_layer, (j + 1) * vae_tile_size + overlap)
                                    
                                    tile_latents = layer_latent[:, :, h_start:h_end, w_start:w_end]
                                    tile_image = decode_tile(tile_latents)
                                    
                                    # Remove overlap
                                    if i > 0:
                                        tile_image = tile_image[:, :, overlap * 8:, :]
                                    if j > 0:
                                        tile_image = tile_image[:, :, :, overlap * 8:]
                                    if i < num_tiles_h - 1 and h_end < H_layer:
                                        remaining_h = (h_end - h_start) * 8 - tile_image.shape[2]
                                        if remaining_h > 0:
                                            tile_image = tile_image[:, :, :-remaining_h, :]
                                    if j < num_tiles_w - 1 and w_end < W_layer:
                                        remaining_w = (w_end - w_start) * 8 - tile_image.shape[3]
                                        if remaining_w > 0:
                                            tile_image = tile_image[:, :, :, :-remaining_w]
                                    
                                    row_tiles.append(tile_image)
                                    del tile_latents
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                
                                row_image = torch.cat(row_tiles, dim=3)
                                decoded_tiles.append(row_image)
                                del row_tiles
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            
                            layer_image = torch.cat(decoded_tiles, dim=2)
                            layer_decoded.append(layer_image)
                            del layer_latent, decoded_tiles
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Concatenate results (keep on GPU for pipeline compatibility)
                        result = torch.cat(layer_decoded, dim=0)
                        del layer_decoded
                    else:
                        # No optimization: use original decode
                        result = original_vae_decode(latents, return_dict=return_dict)
                        # Handle both return_dict=True (DecoderOutput) and return_dict=False (tuple) cases
                        if return_dict:
                            result = result.sample
                        else:
                            # When return_dict=False, VAE decode returns a tuple, take first element
                            if isinstance(result, tuple):
                                result = result[0]
                    
                    # Ensure result is a 4D tensor [B, C, H, W] for pipeline compatibility
                    if len(result.shape) != 4:
                        print(f"[WARNING] VAE decode result shape is {result.shape}, expected 4D [B, C, H, W]", flush=True)
                        # Try to add batch dimension if missing
                        if len(result.shape) == 3:
                            result = result.unsqueeze(0)
                    
                    if return_dict:
                        try:
                            from diffusers.models.vae import DecoderOutput
                            return DecoderOutput(sample=result)
                        except ImportError:
                            # Fallback if DecoderOutput not available
                            class DecoderOutput:
                                def __init__(self, sample):
                                    self.sample = sample
                            return DecoderOutput(sample=result)
                    # When return_dict=False, pipeline expects a tuple (result,)
                    # Pipeline code does: self.vae.decode(...)[0]
                    return (result,)
                else:
                    # Fallback to original decode for non-batched latents
                    # Ensure we return tuple when return_dict=False (pipeline expects [0] indexing)
                    if return_dict:
                        return original_vae_decode(latents, return_dict=return_dict)
                    else:
                        result = original_vae_decode(latents, return_dict=False)
                        # Ensure it's a tuple (pipeline does [0] indexing)
                        if isinstance(result, tuple):
                            return result
                        return (result,)
            
            # Apply monkey patch to VAE instance
            pipeline.vae.decode = optimized_vae_decode_wrapper
            print(f"[INFO] Applied optimized VAE decoding: tiled (tile_size={vae_tile_size})", flush=True)

        # Use PipelineDataset instead of LayoutTrainDataset
        print("[INFO] Loading dataset using PipelineDataset (no HuggingFace download)...", flush=True)
        
        # Resolve data_dir path (relative to repo root or config file location)
        data_dir = config['data_dir']
        data_dir_path = Path(data_dir)
        if not data_dir_path.is_absolute():
            # Try relative to repo root first
            repo_data_dir = repo_root / data_dir
            if repo_data_dir.exists():
                data_dir = str(repo_data_dir.resolve())
            else:
                # Try relative to config file location
                config_data_dir = config_path.parent / data_dir
                if config_data_dir.exists():
                    data_dir = str(config_data_dir.resolve())
                else:
                    # Use as-is and let PipelineDataset handle it
                    data_dir = str(data_dir_path)
        
        print(f"[INFO] Using data_dir: {data_dir}", flush=True)
        
        # Check if directory exists and has JSON files before creating dataset
        data_dir_check = Path(data_dir)
        if not data_dir_check.exists():
            print(f"❌ Error: Data directory does not exist: {data_dir}", flush=True)
            print(f"   Please ensure Step 3 (CLD Format Conversion) has been run first.", flush=True)
            print(f"   Expected directory: {data_dir}", flush=True)
            print(f"   This directory should contain JSON files generated by pipeline_to_cld_infer.py", flush=True)
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        json_files = list(data_dir_check.glob("*.json"))
        if len(json_files) == 0:
            print(f"❌ Error: No JSON files found in {data_dir}", flush=True)
            print(f"   Please ensure Step 3 (CLD Format Conversion) has been run first.", flush=True)
            print(f"   Expected JSON files in: {data_dir}", flush=True)
            print(f"   Found files in directory: {list(data_dir_check.iterdir())}", flush=True)
            raise ValueError(f"No JSON files found in {data_dir}")
        
        dataset = PipelineDataset(
            data_dir=data_dir,
            # DISABLED: max_image_side to preserve image quality
            # max_image_side=config.get('max_image_side'),
            max_image_side=None,  # Disabled to preserve image quality
            max_image_size=config.get('max_image_size'),
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_pipeline)

        generator = torch.Generator(device=torch.device("cuda")).manual_seed(config.get('seed', 42))

        # Get start_from parameter from config or command line args
        # Note: _args_start_from is captured from outer scope (main function)
        start_from = config.get('start_from', 0)
        # Override with command line args if provided
        if _args_start_from > 0:
            start_from = _args_start_from
        
        if start_from > 0:
            print(f"[INFO] Starting from file index {start_from} (skipping first {start_from} files)", flush=True)
            if start_from >= len(dataset):
                print(f"❌ Error: start_from ({start_from}) >= number of files ({len(dataset)})", flush=True)
                raise ValueError(f"start_from ({start_from}) must be less than number of files ({len(dataset)})")

        import gc  # For garbage collection
        import itertools
        
        # Skip first start_from batches if start_from > 0
        batch_iter = itertools.islice(loader, start_from, None) if start_from > 0 else loader
        
        idx = start_from  # Start idx from start_from, so output case_N matches file index N
        for batch in batch_iter:
            print(f"Processing case {idx}", flush=True)
            
            # Clear cache before processing each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            height = int(batch["height"][0])
            width = int(batch["width"][0])
            adapter_img = batch["whole_img"][0]
            caption = batch["caption"][0]
            
            # Pass image dimensions to get_input_box for boundary clamping
            layer_boxes = get_input_box(batch["layout"][0], img_width=width, img_height=height)
            
            # Clear batch from memory before pipeline call
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Generate layers using pipeline
            print(f"[DEBUG] Calling pipeline with validation_box={len(layer_boxes)} boxes...", flush=True)
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
            
            # Debug: Check pipeline output
            print(f"[DEBUG] Pipeline output:", flush=True)
            print(f"  x_hat shape: {x_hat.shape} (expected: [1, num_layers, 4, H, W])", flush=True)
            print(f"  x_hat layers: {x_hat.shape[1] if len(x_hat.shape) > 1 else 'N/A'} (expected: {len(layer_boxes)})", flush=True)
            if isinstance(image, (list, tuple)):
                print(f"  image length: {len(image)} (expected: {len(layer_boxes)})", flush=True)
            else:
                print(f"  image type: {type(image)}", flush=True)

            # Adjust x_hat range from [-1, 1] to [0, 1]
            x_hat = (x_hat + 1) / 2

            # Remove batch dimension and ensure float32 dtype
            # Move to CPU immediately to free GPU memory
            x_hat = x_hat.squeeze(0).permute(1, 0, 2, 3).cpu().to(torch.float32)
            
            # Also move image to CPU
            if isinstance(image, torch.Tensor):
                image = image.cpu()
            elif isinstance(image, (list, tuple)):
                image = [img.cpu() if isinstance(img, torch.Tensor) else img for img in image]
            
            # Delete latents immediately (not needed after this point)
            del latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            this_index = f"case_{idx}"
            case_dir = os.path.join(config['save_dir'], this_index)
            os.makedirs(case_dir, exist_ok=True)
            
            # Save whole image_RGBA (X_hat[0]) and background_RGBA (X_hat[1])
            # x_hat is already on CPU, so no need to call .cpu() again
            whole_image_layer = (x_hat[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Pillow can auto-detect RGBA format from shape [H, W, 4]
            whole_image_rgba_image = Image.fromarray(whole_image_layer)
            whole_image_rgba_image.save(os.path.join(case_dir, "whole_image_rgba.png"))
            del whole_image_layer, whole_image_rgba_image

            adapter_img.save(os.path.join(case_dir, "origin.png"))

            background_layer = (x_hat[1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Pillow can auto-detect RGBA format from shape [H, W, 4]
            background_rgba_image = Image.fromarray(background_layer)
            background_rgba_image.save(os.path.join(case_dir, "background_rgba.png"))
            del background_layer, background_rgba_image

            x_hat = x_hat[2:]
            merged_image = image[1]
            image = image[2:]

            # Save transparent VAE decoded results
            # x_hat[2:] corresponds to foreground layers (after whole_image and background)
            print(f"[DEBUG] Saving {x_hat.shape[0]} foreground layers...", flush=True)
            for layer_idx in range(x_hat.shape[0]):
                layer = x_hat[layer_idx]  # Already on CPU, shape: [4, H, W] (RGBA)
                rgba_layer = (layer.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                
                
                # Pillow can auto-detect RGBA format from shape [H, W, 4]
                rgba_image = Image.fromarray(rgba_layer, mode='RGBA')
                rgba_image.save(os.path.join(case_dir, f"layer_{layer_idx}_rgba.png"))
                # Clean up immediately after saving
                del layer, rgba_layer, rgba_image

            # Composite background and foreground layers
            for layer_idx in range(x_hat.shape[0]):
                rgba_layer = (x_hat[layer_idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                layer_image = Image.fromarray(rgba_layer, mode='RGBA')
                merged_image = Image.alpha_composite(merged_image.convert('RGBA'), layer_image)
                # Clean up immediately
                del rgba_layer, layer_image
            
            # Save final composite images
            merged_image.convert('RGB').save(os.path.join(config['save_dir'], "merged", f"{this_index}.png"))
            merged_image.convert('RGB').save(os.path.join(case_dir, f"{this_index}.png"))
            # Save final composite RGBA image
            merged_image.save(os.path.join(config['save_dir'], "merged_rgba", f"{this_index}.png"))

            print(f"Saved case {idx} to {case_dir}")
            
            # Aggressive VRAM cleanup after each image
            # Delete all intermediate variables
            try:
                del x_hat, image, whole_image_layer, background_layer, rgba_layer, layer_image, merged_image
                del whole_image_rgba_image, background_rgba_image, rgba_image, layer_image
            except NameError:
                pass
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Try to reclaim reserved memory
                torch.cuda.reset_peak_memory_stats()
            
            # Print memory usage every image (for debugging)
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
            
            idx += 1

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    inference_layout_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


