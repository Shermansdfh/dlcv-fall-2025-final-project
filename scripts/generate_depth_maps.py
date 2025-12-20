"""
Step 1: Generate depth maps for DLCV dataset using ml-depth-pro.
This script should be run in the depth-pro conda environment.

Usage:
    conda activate depth-pro
    python scripts/generate_depth_maps.py --max_samples 20000 --output_dir depth_exp/depth_maps
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# Add ml-depth-pro to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DEPTH_PRO_PATH = PROJECT_ROOT / "depth_exp" / "ml-depth-pro"
if str(ML_DEPTH_PRO_PATH / "src") not in sys.path:
    sys.path.insert(0, str(ML_DEPTH_PRO_PATH / "src"))

try:
    import depth_pro
except ImportError:
    print("ERROR: depth_pro is not available. Please activate the depth-pro conda environment:")
    print("  conda activate depth-pro")
    print("  pip install -e depth_exp/ml-depth-pro")
    sys.exit(1)

from datasets import load_dataset


def generate_depth_for_image(model, transform, image, img_id, output_path):
    """Generate and save depth map for a single image."""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Transform image for depth model
        img_tensor = transform(img_array)
        
        # Run inference
        with torch.no_grad():
            prediction = model.infer(img_tensor, f_px=None)
            depth = prediction["depth"]  # Depth in meters
        
        # Convert to numpy
        depth_np = depth.detach().cpu().numpy()
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze()
        elif depth_np.ndim == 4:
            depth_np = depth_np.squeeze(0).squeeze(0)
        
        # Normalize depth to [0, 1] range (same as dataset)
        depth_clipped = np.clip(depth_np, 0.1, 100.0)
        depth_normalized = (depth_clipped - 0.1) / (100.0 - 0.1)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as npz (compressed numpy format)
        np.savez_compressed(output_path, depth=depth_normalized.astype(np.float32))
        
        return True
    except Exception as e:
        print(f"  WARNING: Failed to generate depth for {img_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for DLCV dataset")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="depth_exp/depth_maps",
        help="Output directory for depth maps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling"
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=2000,
        help="Buffer size for streaming shuffle"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for depth model (cuda/cpu, None for auto)"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"[INFO] Using device: {device}")
    
    # Check and fix checkpoint path
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig
    
    # Try to find checkpoint in ml-depth-pro directory
    checkpoint_path = ML_DEPTH_PRO_PATH / "checkpoints" / "depth_pro.pt"
    
    if not checkpoint_path.exists():
        # Try relative path from ml-depth-pro directory
        checkpoint_path_rel = Path("./checkpoints/depth_pro.pt")
        if checkpoint_path_rel.exists():
            checkpoint_path = checkpoint_path_rel.resolve()
        else:
            print(f"[ERROR] Checkpoint file not found!")
            print(f"  Tried: {checkpoint_path}")
            print(f"  Tried: {checkpoint_path_rel.resolve()}")
            print(f"\n[INFO] Please download the checkpoint first:")
            print(f"  cd depth_exp/ml-depth-pro")
            print(f"  source get_pretrained_models.sh")
            print(f"\nOr manually download from:")
            print(f"  https://github.com/apple/ml-depth-pro")
            sys.exit(1)
    
    print(f"[INFO] Using checkpoint: {checkpoint_path}")
    
    # Create custom config with correct checkpoint path
    custom_config = DepthProConfig(
        patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
        image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
        decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
        checkpoint_uri=str(checkpoint_path),
        use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
        fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
    )
    
    # Load depth model
    print("[INFO] Loading depth model...")
    model, transform = depth_pro.create_model_and_transforms(
        config=custom_config,
        device=device,
        precision=torch.half if device.type == "cuda" else torch.float32,
    )
    model.eval()
    print("[INFO] Depth model loaded successfully")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Load dataset
    print("[INFO] Loading DLCV dataset...")
    dataset = load_dataset(
        "WalkerHsu/DLCV2025_final_project_piccollage",
        split="train",
        streaming=True
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    
    # Process samples
    count = 0
    success_count = 0
    failed_count = 0
    
    print(f"[INFO] Starting depth map generation...")
    for sample in tqdm(dataset, total=args.max_samples, desc="Generating depth maps"):
        if args.max_samples is not None and count >= args.max_samples:
            break
        
        # Check if sample is valid
        if 'preview' not in sample or sample['preview'] is None:
            continue
        
        if 'id' not in sample:
            continue
        
        img_id = sample['id']
        main_img = sample['preview'].convert("RGB")
        
        # Generate output path
        # Normalize img_id to get numeric part
        img_id_str = str(img_id)
        try:
            img_id_num = int(Path(img_id_str).stem)
            img_id_padded = f"{img_id_num:08d}"
        except ValueError:
            img_id_padded = Path(img_id_str).stem
        
        output_path = output_dir / f"{img_id_padded}.npz"
        
        # Skip if already exists
        if output_path.exists():
            count += 1
            continue
        
        # Generate depth map
        success = generate_depth_for_image(
            model, transform, main_img, img_id, output_path
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        count += 1
    
    print(f"\n[INFO] Generation completed!")
    print(f"  - Total processed: {count}")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Output directory: {output_dir}")
    print(f"\n[INFO] Next step: Use depth_map_dir in train_dlcv.yaml to load these depth maps")


if __name__ == "__main__":
    main()

