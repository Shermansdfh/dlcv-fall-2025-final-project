"""
LayerD Preprocessing: Generate LayerD masks for `LayerDecompositionPipeline`.

Based on `LayerD/tools/infer.py` and `layerd.models.layerd.LayerD`:
- Load LayerD
- Decompose a single image
- Extract alpha channel as masks, form Front-to-Back, with the last element being the background
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LAYERD_DIR = REPO_ROOT / "third_party" / "layerd"
sys.path.append(str(LAYERD_DIR))

from layerd.models.layerd import LayerD

@dataclass
class LayerDMasks:
    """LayerD masks for `LayerDecompositionPipeline`."""

    image_path: str
    image_size: Tuple[int, int]  # (H, W)
    masks: List[np.ndarray]  # Front-to-Back, with the last element being the background


def load_layerd(
    matting_hf_card: str = "cyberagent/layerd-birefnet",
    matting_process_size: Optional[Tuple[int, int]] = None,
    matting_weight_path: Optional[str] = None,
    kernel_scale: float = 0.015,
    device: str = "cpu",
) -> LayerD:
    """Load LayerD model."""
    return LayerD(
        matting_hf_card=matting_hf_card,
        matting_process_size=matting_process_size,
        matting_weight_path=matting_weight_path,
        kernel_scale=kernel_scale,
        device=device,
    )


def _rgba_to_alpha(mask_rgba: Image.Image) -> np.ndarray:
    """Convert RGBA image to 0~1 alpha mask (float)."""
    if mask_rgba.mode != "RGBA":
        mask_rgba = mask_rgba.convert("RGBA")
    alpha = np.array(mask_rgba.split()[-1], dtype=float) / 255.0
    return alpha


def decompose_to_masks(
    layerd: LayerD,
    image_path: str,
    max_iterations: int = 3,
    device: str = "cpu",
    max_image_size: Optional[Tuple[int, int]] = None,
) -> LayerDMasks:
    """
    Decompose a single image with LayerD, return Front-to-Back masks, with the last element being the background.
    
    Args:
        max_image_size: Optional (width, height) to resize large images before processing to save memory.
    """
    import torch
    import gc
    
    img = Image.open(image_path).convert("RGB")
    original_size = (img.height, img.width)
    was_resized = False
    
    # Resize large images to reduce memory usage (especially for LaMa model)
    # Note: This affects quality - masks are upscaled back but may lose fine details
    # For best quality, set max_image_size to null or a very large value
    if max_image_size is not None:
        max_w, max_h = max_image_size
        if img.width > max_w or img.height > max_h:
            # Calculate scaling factor to fit within max_image_size while maintaining aspect ratio
            scale = min(max_w / img.width, max_h / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            was_resized = True
            reduction_pct = (1 - scale) * 100
            print(f"  ⚠️  Resized image from {original_size[1]}x{original_size[0]} to {new_size[0]}x{new_size[1]} ({reduction_pct:.1f}% reduction) to save memory. Quality may be affected.")
    
    # Aggressively clear cache before decomposition
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        gc.collect()
    
    # Use torch.no_grad() to prevent gradient computation and reduce memory
    with torch.no_grad():
        layers = layerd.decompose(img, max_iterations=max_iterations)
    
    # Aggressively clear cache immediately after decomposition
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()  # Wait for all operations to complete
        torch.cuda.empty_cache()
        # Clear TorchScript JIT cache (LaMa model uses JIT)
        try:
            torch.jit.clear_class_registry()
        except Exception:
            pass
        gc.collect()
        # Force another cache clear after GC
        torch.cuda.empty_cache()
    
    # LayerD.decompose returns [background, fg... (from back to front)], need to convert to Front-to-Back, with the last element being the background
    background = layers[0]
    fgs = layers[1:]
    
    # Use original image size (not resized size) for output
    image_size = original_size

    fg_masks = [_rgba_to_alpha(fg) for fg in fgs]  # assume LayerD output is from front to back
    bg_mask = _rgba_to_alpha(background)
    
    # If image was resized, resize masks back to original size
    # Use cubic interpolation for better quality when upscaling
    if was_resized:
        import cv2
        target_size = (original_size[1], original_size[0])  # (width, height) for cv2
        # Save current (resized) image size before deleting
        resized_size = (img.height, img.width)
        # Use INTER_CUBIC for upscaling (better quality than LINEAR, but slower)
        # For downscaling, LINEAR is sufficient
        interpolation = cv2.INTER_CUBIC if (original_size[1] > resized_size[1] or original_size[0] > resized_size[0]) else cv2.INTER_LINEAR
        fg_masks = [cv2.resize(mask, target_size, interpolation=interpolation) for mask in fg_masks]
        bg_mask = cv2.resize(bg_mask, target_size, interpolation=interpolation)
    
    # Explicitly delete layers to free memory
    del layers, background, fgs, img
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Clear TorchScript JIT cache
        try:
            torch.jit.clear_class_registry()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    # form Front-to-Back, with the last element being the background
    masks: List[np.ndarray] = list(fg_masks) + [bg_mask]

    return LayerDMasks(
        image_path=image_path,
        image_size=image_size,
        masks=masks,
    )


def batch_decompose(
    layerd: LayerD,
    sources: Sequence[str],
    max_iterations: int = 3,
    limit: Optional[int] = None,
    device: str = "cpu",
    max_image_size: Optional[Tuple[int, int]] = None,
) -> List[LayerDMasks]:
    """Decompose multiple images with LayerD, get masks list."""
    import torch
    import gc
    
    outputs: List[LayerDMasks] = []
    total = len(sources) if limit is None else min(len(sources), limit)
    
    for idx, image_path in enumerate(sources):
        if limit is not None and idx >= limit:
            break
        try:
            # Aggressively clear cache before processing each image
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
            
            result = decompose_to_masks(layerd, image_path, max_iterations=max_iterations, device=device, max_image_size=max_image_size)
            outputs.append(result)
            
            # Aggressively clear GPU cache and Python garbage after each image to prevent memory accumulation
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Clear TorchScript JIT cache (LaMa model uses JIT)
                try:
                    torch.jit.clear_class_registry()
                except Exception:
                    pass
                gc.collect()
                # Force another cache clear after GC
                torch.cuda.empty_cache()
            
            # Print progress with memory info
            if torch.cuda.is_available() and device == "cuda":
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                # Get actual GPU memory usage from nvidia-smi (more accurate than PyTorch tracking)
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        actual_gpu_mem = float(result.stdout.strip()) / 1024
                        print(f"  Processed {idx + 1}/{total}: {Path(image_path).name} (PyTorch: {mem_alloc:.2f}GB/{mem_reserved:.2f}GB, Actual GPU: {actual_gpu_mem:.2f}GB)")
                    else:
                        print(f"  Processed {idx + 1}/{total}: {Path(image_path).name} (PyTorch: {mem_alloc:.2f}GB/{mem_reserved:.2f}GB)")
                except Exception:
                    print(f"  Processed {idx + 1}/{total}: {Path(image_path).name} (PyTorch: {mem_alloc:.2f}GB/{mem_reserved:.2f}GB)")
            else:
                print(f"  Processed {idx + 1}/{total}: {Path(image_path).name}")
        except Exception as e:
            print(f"  ❌ Failed to process {image_path}: {e}")
            # Still clear cache even on error
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            raise
    
    return outputs


def load_rtdetr_results(rtdetr_output_dir: str) -> dict:
    """Load RTDETR results from JSON files."""
    import json
    from pathlib import Path
    
    results = {}
    rtdetr_path = Path(rtdetr_output_dir)
    for json_file in rtdetr_path.glob("*.json"):
        stem = json_file.stem
        data = json.loads(json_file.read_text(encoding="utf-8"))
        results[stem] = data
    return results


def save_layerd_results(masks_list: List[LayerDMasks], output_dir: str):
    """Save LayerD masks to NPZ files."""
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for masks in masks_list:
        stem = Path(masks.image_path).stem
        npz_path = output_path / f"{stem}.npz"
        # Convert list of masks to numpy array for storage
        masks_array = np.array(masks.masks, dtype=object)
        np.savez_compressed(
            npz_path,
            masks=masks_array,
            image_size=np.array(masks.image_size),
            image_path=np.array(masks.image_path, dtype=object),
        )
    
    print(f"✅ Saved {len(masks_list)} LayerD results to {output_dir}")


def resolve_path(path: str, config_path: Path) -> Path:
    """Resolve relative paths relative to config file location."""
    p = Path(path)
    if p.is_absolute():
        return p
    # Resolve relative to config file's parent directory
    return (config_path.parent / p).resolve()


if __name__ == "__main__":
    import argparse
    import os
    import yaml
    from pathlib import Path
    
    # Set PyTorch memory allocator to use expandable segments to reduce fragmentation
    # This helps prevent OOM errors when processing multiple images
    # Use PYTORCH_ALLOC_CONF (new) instead of PYTORCH_CUDA_ALLOC_CONF (deprecated)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # LayerD config - resolve paths relative to config file
    rtdetr_output_dir = resolve_path(config["layerd"]["rtdetr_output_dir"], config_path)
    output_dir = resolve_path(config["layerd"]["output_dir"], config_path)
    max_iterations = config["layerd"].get("max_iterations", 3)
    device = config["layerd"].get("device", "cpu")
    limit = config["layerd"].get("limit", None)
    
    # Optional: matting_process_size to limit image processing size for memory management
    # Format: [width, height] or null for no limit (uses model default, typically 1024x1024)
    matting_process_size = config["layerd"].get("matting_process_size", None)
    if matting_process_size is not None:
        matting_process_size = tuple(matting_process_size)  # Convert list to tuple
    
    # Optional: max_image_size to resize large images before processing (reduces LaMa memory usage)
    # Format: [width, height] or null for no limit
    max_image_size = config["layerd"].get("max_image_size", None)
    if max_image_size is not None:
        max_image_size = tuple(max_image_size)  # Convert list to tuple
    
    # Load RTDETR results to get image paths
    rtdetr_results = load_rtdetr_results(str(rtdetr_output_dir))
    if not rtdetr_results:
        print(f"❌ No RTDETR results found in {rtdetr_output_dir}")
        exit(1)
    
    image_paths = [r["image_path"] for r in rtdetr_results.values()]
    
    # Clear GPU cache before loading model
    import torch
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"💾 GPU memory before model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    print(f"🚀 Loading LayerD model...")
    if matting_process_size:
        print(f"   Using processing size: {matting_process_size} (to reduce memory usage)")
    layerd = load_layerd(device=device, matting_process_size=matting_process_size)
    
    # Clear GPU cache after loading model
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"💾 GPU memory after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    print(f"📂 Processing {len(image_paths) if limit is None else min(len(image_paths), limit)} images...")
    if max_image_size:
        print(f"   Using max image size: {max_image_size} (large images will be resized to save memory)")
    masks_list = batch_decompose(layerd, image_paths, max_iterations=max_iterations, limit=limit, device=device, max_image_size=max_image_size)
    
    save_layerd_results(masks_list, str(output_dir))


