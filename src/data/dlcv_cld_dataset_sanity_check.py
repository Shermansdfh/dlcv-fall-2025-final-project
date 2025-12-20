"""
Sanity check script for DLCVCLDDataset.
This script verifies that the dataset loads correctly and produces valid data for CLD training.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dlcv_cld_dataset import DLCVCLDDataset, collate_fn


def check_dataset_basic(dataset, num_samples=5):
    """Check basic dataset functionality."""
    print("\n" + "="*60)
    print("1. Basic Dataset Check")
    print("="*60)
    
    # Check dataset length
    dataset_len = len(dataset)
    print(f"✓ Dataset length: {dataset_len}")
    
    if dataset_len == 0:
        print("✗ ERROR: Dataset is empty!")
        return False
    
    # Check sample access
    print(f"\nChecking first {min(num_samples, dataset_len)} samples...")
    for i in range(min(num_samples, dataset_len)):
        try:
            sample = dataset[i]
            print(f"  ✓ Sample {i}: Loaded successfully")
        except Exception as e:
            print(f"  ✗ Sample {i}: Failed - {e}")
            return False
    
    return True


def check_sample_format(sample, sample_idx=0):
    """Check if sample has correct format."""
    print("\n" + "="*60)
    print(f"2. Sample Format Check (Sample {sample_idx})")
    print("="*60)
    
    required_keys = ["pixel_RGBA", "pixel_RGB", "whole_img", "caption", "height", "width", "layout"]
    missing_keys = [key for key in required_keys if key not in sample]
    
    if missing_keys:
        print(f"✗ ERROR: Missing keys: {missing_keys}")
        return False
    
    print(f"✓ All required keys present: {required_keys}")
    
    # Check pixel_RGBA
    if not isinstance(sample["pixel_RGBA"], list):
        print(f"✗ ERROR: pixel_RGBA should be a list, got {type(sample['pixel_RGBA'])}")
        return False
    
    num_layers = len(sample["pixel_RGBA"])
    print(f"✓ Number of layers: {num_layers}")
    
    if num_layers < 2:
        print(f"✗ WARNING: Expected at least 2 layers (whole image + base), got {num_layers}")
    
    # Check each layer tensor
    for i, layer_tensor in enumerate(sample["pixel_RGBA"]):
        if not isinstance(layer_tensor, torch.Tensor):
            print(f"✗ ERROR: Layer {i} should be torch.Tensor, got {type(layer_tensor)}")
            return False
        
        if layer_tensor.dim() != 3:
            print(f"✗ ERROR: Layer {i} should be 3D tensor [C, H, W], got shape {layer_tensor.shape}")
            return False
        
        C, H, W = layer_tensor.shape
        if C != 4:
            print(f"✗ WARNING: Layer {i} should have 4 channels (RGBA), got {C}")
        
        print(f"  ✓ Layer {i}: shape {layer_tensor.shape}, dtype {layer_tensor.dtype}, range [{layer_tensor.min():.3f}, {layer_tensor.max():.3f}]")
    
    # Check pixel_RGB
    if len(sample["pixel_RGB"]) != num_layers:
        print(f"✗ ERROR: pixel_RGB should have {num_layers} layers, got {len(sample['pixel_RGB'])}")
        return False
    
    for i, layer_tensor in enumerate(sample["pixel_RGB"]):
        if not isinstance(layer_tensor, torch.Tensor):
            print(f"✗ ERROR: RGB Layer {i} should be torch.Tensor, got {type(layer_tensor)}")
            return False
        
        C, H, W = layer_tensor.shape
        if C != 3:
            print(f"✗ WARNING: RGB Layer {i} should have 3 channels, got {C}")
    
    print(f"✓ pixel_RGB: {len(sample['pixel_RGB'])} layers")
    
    # Check whole_img
    if not isinstance(sample["whole_img"], Image.Image):
        print(f"✗ ERROR: whole_img should be PIL.Image, got {type(sample['whole_img'])}")
        return False
    
    img_w, img_h = sample["whole_img"].size
    print(f"✓ whole_img: PIL.Image, size {img_w}x{img_h}, mode {sample['whole_img'].mode}")
    
    # Check caption
    if not isinstance(sample["caption"], str):
        print(f"✗ ERROR: caption should be str, got {type(sample['caption'])}")
        return False
    
    caption_len = len(sample["caption"])
    print(f"✓ caption: str, length {caption_len}")
    if caption_len > 0:
        print(f"  Preview: '{sample['caption'][:100]}...'")
    else:
        print(f"  WARNING: Caption is empty")
    
    # Check height and width
    if not isinstance(sample["height"], int) or not isinstance(sample["width"], int):
        print(f"✗ ERROR: height/width should be int, got {type(sample['height'])}/{type(sample['width'])}")
        return False
    
    if sample["height"] != img_h or sample["width"] != img_w:
        print(f"✗ WARNING: height/width mismatch: ({sample['width']}, {sample['height']}) vs image size ({img_w}, {img_h})")
    
    print(f"✓ height: {sample['height']}, width: {sample['width']}")
    
    # Check layout
    if not isinstance(sample["layout"], list):
        print(f"✗ ERROR: layout should be a list, got {type(sample['layout'])}")
        return False
    
    if len(sample["layout"]) != num_layers:
        print(f"✗ ERROR: layout should have {num_layers} boxes, got {len(sample['layout'])}")
        return False
    
    for i, box in enumerate(sample["layout"]):
        if not isinstance(box, list) or len(box) != 4:
            print(f"✗ ERROR: Layout box {i} should be [x1, y1, x2, y2], got {box}")
            return False
        
        x1, y1, x2, y2 = box
        if not all(isinstance(coord, (int, float)) for coord in box):
            print(f"✗ ERROR: Layout box {i} coordinates should be numeric, got {box}")
            return False
        
        if x1 >= x2 or y1 >= y2:
            print(f"✗ WARNING: Layout box {i} has invalid coordinates: {box}")
        
        # Check if box is within image bounds
        if x1 < 0 or y1 < 0 or x2 > sample["width"] or y2 > sample["height"]:
            print(f"  WARNING: Layout box {i} extends beyond image bounds: {box} vs image {sample['width']}x{sample['height']}")
    
    print(f"✓ layout: {len(sample['layout'])} boxes")
    print(f"  First box: {sample['layout'][0]}")
    print(f"  Last box: {sample['layout'][-1]}")
    
    return True


def check_caption_loading(dataset, num_samples=5):
    """Check if captions are loaded correctly."""
    print("\n" + "="*60)
    print("3. Caption Loading Check")
    print("="*60)
    
    captions_found = 0
    captions_empty = 0
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        caption = sample["caption"]
        
        if len(caption) > 0:
            captions_found += 1
            print(f"  ✓ Sample {i}: Caption found ({len(caption)} chars)")
        else:
            captions_empty += 1
            print(f"  ⚠ Sample {i}: Caption is empty")
    
    print(f"\nSummary: {captions_found}/{num_samples} samples have captions")
    
    if captions_found == 0:
        print("✗ WARNING: No captions found! Check caption_llava15.json")
        return False
    
    return True


def check_collate_fn(dataset, batch_size=2):
    """Check if collate function works correctly."""
    print("\n" + "="*60)
    print("4. Collate Function Check")
    print("="*60)
    
    if len(dataset) < batch_size:
        print(f"⚠ Skipping: Dataset has only {len(dataset)} samples, need {batch_size} for batch test")
        return True
    
    # Create a small batch
    samples = [dataset[i] for i in range(batch_size)]
    
    try:
        batch = collate_fn(samples)
        print(f"✓ Collate function executed successfully")
    except Exception as e:
        print(f"✗ ERROR: Collate function failed - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check batch structure
    required_keys = ["pixel_RGBA", "pixel_RGB", "whole_img", "caption", "height", "width", "layout"]
    missing_keys = [key for key in required_keys if key not in batch]
    
    if missing_keys:
        print(f"✗ ERROR: Batch missing keys: {missing_keys}")
        return False
    
    # Check batch shapes
    pixel_RGBA = batch["pixel_RGBA"]
    pixel_RGB = batch["pixel_RGB"]
    
    if not isinstance(pixel_RGBA, torch.Tensor):
        print(f"✗ ERROR: batch['pixel_RGBA'] should be torch.Tensor, got {type(pixel_RGBA)}")
        return False
    
    if pixel_RGBA.dim() != 5:
        print(f"✗ ERROR: batch['pixel_RGBA'] should be 5D [B, L, C, H, W], got shape {pixel_RGBA.shape}")
        return False
    
    B, L, C, H, W = pixel_RGBA.shape
    if B != batch_size:
        print(f"✗ ERROR: Batch size mismatch: expected {batch_size}, got {B}")
        return False
    
    print(f"✓ batch['pixel_RGBA']: shape {pixel_RGBA.shape}, dtype {pixel_RGBA.dtype}")
    print(f"✓ batch['pixel_RGB']: shape {pixel_RGB.shape}, dtype {pixel_RGB.dtype}")
    print(f"✓ batch['whole_img']: {len(batch['whole_img'])} images")
    print(f"✓ batch['caption']: {len(batch['caption'])} captions")
    print(f"✓ batch['height']: {len(batch['height'])} values")
    print(f"✓ batch['width']: {len(batch['width'])} values")
    print(f"✓ batch['layout']: {len(batch['layout'])} layouts")
    
    return True


def check_dataloader(dataset, batch_size=1, num_batches=3):
    """Check if DataLoader works correctly."""
    print("\n" + "="*60)
    print("5. DataLoader Check")
    print("="*60)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"✓ DataLoader created: batch_size={batch_size}, num_workers=0")
    
    try:
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            print(f"  ✓ Batch {i+1}: Loaded successfully")
            print(f"    - pixel_RGBA: {batch['pixel_RGBA'].shape}")
            print(f"    - pixel_RGB: {batch['pixel_RGB'].shape}")
            print(f"    - Captions: {[len(c) for c in batch['caption']]}")
        
        print(f"✓ DataLoader iteration successful ({num_batches} batches)")
        return True
    except Exception as e:
        print(f"✗ ERROR: DataLoader iteration failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def check_depth_feature(dataset):
    """Check if depth feature is available and working."""
    print("\n" + "="*60)
    print("6. Depth Feature Check")
    print("="*60)
    
    if not dataset.use_depth:
        print("ℹ Depth feature is disabled (use_depth=False)")
        return True
    
    if dataset.depth_model is None:
        print("✗ WARNING: use_depth=True but depth_model is None")
        return False
    
    print("✓ Depth feature is enabled")
    print(f"  - Depth device: {dataset.depth_device}")
    print(f"  - Depth model loaded: {dataset.depth_model is not None}")
    
    # Try to generate depth for a sample
    if len(dataset) > 0:
        sample = dataset[0]
        whole_img = sample["whole_img"]
        
        try:
            depth_map = dataset._generate_depth(whole_img)
            if depth_map is not None:
                print(f"✓ Depth generation successful: shape {depth_map.shape}, range [{depth_map.min():.3f}, {depth_map.max():.3f}]")
                return True
            else:
                print("✗ WARNING: Depth generation returned None")
                return False
        except Exception as e:
            print(f"✗ ERROR: Depth generation failed - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def main():
    """Run all sanity checks."""
    print("="*60)
    print("DLCV CLD Dataset Sanity Check")
    print("="*60)
    
    # Test 1: Basic dataset without depth
    print("\n[TEST 1] Basic Dataset (no depth)")
    try:
        dataset_basic = DLCVCLDDataset(
            split="train",
            max_samples=10,
            use_depth=False
        )
        
        if not check_dataset_basic(dataset_basic, num_samples=5):
            print("\n✗ Basic dataset check failed!")
            return False
        
        if not check_sample_format(dataset_basic[0], sample_idx=0):
            print("\n✗ Sample format check failed!")
            return False
        
        if not check_caption_loading(dataset_basic, num_samples=5):
            print("\n⚠ Caption loading check had warnings")
        
        if not check_collate_fn(dataset_basic, batch_size=2):
            print("\n✗ Collate function check failed!")
            return False
        
        if not check_dataloader(dataset_basic, batch_size=1, num_batches=3):
            print("\n✗ DataLoader check failed!")
            return False
        
        print("\n✓ [TEST 1] PASSED: Basic dataset works correctly")
    except Exception as e:
        print(f"\n✗ [TEST 1] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Dataset with depth (if available)
    print("\n" + "="*60)
    print("[TEST 2] Dataset with Depth Feature")
    print("="*60)
    
    try:
        dataset_depth = DLCVCLDDataset(
            split="train",
            max_samples=3,  # Use fewer samples for depth test
            use_depth=True
        )
        
        if not check_depth_feature(dataset_depth):
            print("\n⚠ [TEST 2] Depth feature check had warnings (this is OK if depth_pro is not installed)")
        else:
            print("\n✓ [TEST 2] PASSED: Depth feature works correctly")
    except ImportError:
        print("ℹ [TEST 2] SKIPPED: depth_pro not available (this is OK)")
    except Exception as e:
        print(f"\n⚠ [TEST 2] WARNING: {e} (this is OK if depth_pro is not installed)")
    
    # Summary
    print("\n" + "="*60)
    print("Sanity Check Summary")
    print("="*60)
    print("✓ All critical checks passed!")
    print("\nThe dataset is ready for CLD training.")
    print("\nNext steps:")
    print("  1. Configure train_dlcv.yaml with your model paths")
    print("  2. Run: python depth_exp/CLD/train/train_dlcv.py -c train_dlcv.yaml")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

