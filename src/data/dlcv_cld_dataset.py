"""
Custom Dataset for CLD training using DLCV2025_final_project_piccollage dataset.
This dataset reads from the same source as dlcv_bbox_dataset.py but formats data for CLD training.
"""

import os
import math
import random
import json
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import List, Dict, Any, Optional, Tuple, Union
import sys

# Add ml-depth-pro to path
ML_DEPTH_PRO_PATH = Path(__file__).resolve().parent.parent.parent / "depth_exp" / "ml-depth-pro"
if str(ML_DEPTH_PRO_PATH) not in sys.path:
    sys.path.insert(0, str(ML_DEPTH_PRO_PATH / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ID = "WalkerHsu/DLCV2025_final_project_piccollage"
CAPTION_JSON_PATH = REPO_ROOT / "depth_exp" / "caption_llava15.json"

# Try to import depth_pro (optional)
try:
    import depth_pro
    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False
    depth_pro = None


def collate_fn(batch):
    """Collate function for CLD training dataset."""
    pixels_RGBA = [torch.stack(item["pixel_RGBA"]) for item in batch]  # [L, C, H, W]
    pixels_RGB  = [torch.stack(item["pixel_RGB"])  for item in batch]  # [L, C, H, W]
    pixels_RGBA = torch.stack(pixels_RGBA)  # [B, L, C, H, W]
    pixels_RGB  = torch.stack(pixels_RGB)   # [B, L, C, H, W]

    return {
        "pixel_RGBA": pixels_RGBA,
        "pixel_RGB": pixels_RGB,
        "whole_img": [item["whole_img"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "height": [item["height"] for item in batch],
        "width": [item["width"] for item in batch],
        "layout": [item["layout"] for item in batch],
    }


# === Core tool 1: Pure geometry rotation (for Angle != 0) ===
def calculate_rotated_aabb(left, top, w, h, angle_deg):
    """
    Calculate Axis-Aligned Bounding Box (AABB) based on the center point rotation
    """
    # 1. Convert to radians
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 2. Find the center point (assuming left/top is the top-left corner before rotation)
    cx = left + w / 2
    cy = top + h / 2

    # 3. Define the four corners relative to the center (before rotation)
    # Top-left, Top-right, Bottom-right, Bottom-left
    corners = [
        (-w / 2, -h / 2),
        ( w / 2, -h / 2),
        ( w / 2,  h / 2),
        (-w / 2,  h / 2)
    ]

    # 4. Rotation transformation
    rotated_xs = []
    rotated_ys = []
    
    for dx, dy in corners:
        # Rotation matrix formula
        nx = cx + (dx * cos_t - dy * sin_t)
        ny = cy + (dx * sin_t + dy * cos_t)
        rotated_xs.append(nx)
        rotated_ys.append(ny)

    # 5. Find the horizontal bounding box
    min_x = min(rotated_xs)
    max_x = max(rotated_xs)
    min_y = min(rotated_ys)
    max_y = max(rotated_ys)

    return min_x, min_y, max_x, max_y


# === Core tool 2: Alpha Crop + scale (for Angle == 0) ===
def get_tight_box_with_scale(layer_asset, meta_left, meta_top, meta_w, meta_h):
    if layer_asset is None: return None
    
    asset_w, asset_h = layer_asset.size
    if asset_w == 0 or asset_h == 0: return None

    # Get the non-transparent region (original pixels)
    alpha_bbox = layer_asset.getbbox()
    if alpha_bbox is None: return None
    a_left, a_top, a_right, a_bottom = alpha_bbox
    
    # Calculate the scale ratio (Canvas Size / Asset Size)
    scale_x = meta_w / asset_w
    scale_y = meta_h / asset_h
    
    # Project to canvas coordinates
    final_x1 = meta_left + (a_left * scale_x)
    final_y1 = meta_top + (a_top * scale_y)
    final_x2 = meta_left + (a_right * scale_x)
    final_y2 = meta_top + (a_bottom * scale_y)
    
    return final_x1, final_y1, final_x2, final_y2


def rgba2rgb(img_RGBA):
    """Convert RGBA image to RGB with gray background."""
    img_RGB = Image.new("RGB", img_RGBA.size, (128, 128, 128))
    img_RGB.paste(img_RGBA, mask=img_RGBA.split()[3])
    return img_RGB


class DLCVCLDDataset(Dataset):
    """
    Dataset for CLD training using DLCV2025_final_project_piccollage dataset.
    
    This dataset loads data from the same source as dlcv_bbox_dataset.py but formats
    it for CLD training, which requires:
    - pixel_RGBA: List of RGBA tensors for each layer
    - pixel_RGB: List of RGB tensors for each layer
    - whole_img: Complete RGB image
    - caption: Text caption (empty string for this dataset)
    - layout: List of bounding boxes [[x1, y1, x2, y2], ...]
    """
    
    def __init__(
        self,
        split: str = "train",
        seed: int = 42,
        max_samples: Optional[int] = None,
        shuffle_buffer_size: int = 2000,
        caption_json_path: Optional[Union[str, Path]] = None,
        use_depth: bool = False,
        depth_device: Optional[torch.device] = None,
    ):
        """
        Initialize dataset from HuggingFace dataset.
        
        Args:
            split: "train" or "val"
            seed: Random seed for shuffling
            max_samples: Maximum number of samples to load (None for all)
            shuffle_buffer_size: Buffer size for streaming shuffle
            caption_json_path: Path to caption_llava15.json file (default: depth_exp/caption_llava15.json)
            use_depth: Whether to add depth channel using ml-depth-pro (default: False)
            depth_device: Device for depth model (default: cuda if available, else cpu)
        """
        self.split = split
        self.seed = seed
        self.max_samples = max_samples
        self.to_tensor = T.ToTensor()
        self.use_depth = use_depth
        
        # Initialize depth model if requested
        self.depth_model = None
        self.depth_transform = None
        if use_depth:
            if not DEPTH_PRO_AVAILABLE:
                raise ImportError(
                    "depth_pro is not available. Please install ml-depth-pro:\n"
                    "  cd depth_exp/ml-depth-pro\n"
                    "  pip install -e ."
                )
            
            if depth_device is None:
                depth_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"[INFO] Loading depth model (device: {depth_device})...")
            try:
                self.depth_model, self.depth_transform = depth_pro.create_model_and_transforms(
                    device=depth_device,
                    precision=torch.half if depth_device.type == "cuda" else torch.float32,
                )
                self.depth_model.eval()
                self.depth_device = depth_device
                print("[INFO] Depth model loaded successfully")
            except Exception as e:
                print(f"[WARN] Failed to load depth model: {e}")
                print("[WARN] Continuing without depth channel")
                self.use_depth = False
                self.depth_model = None
                self.depth_transform = None
        
        # Load caption JSON file
        if caption_json_path is None:
            caption_json_path = CAPTION_JSON_PATH
        else:
            caption_json_path = Path(caption_json_path)
        
        self.caption_dict = {}
        if caption_json_path.exists():
            print(f"[INFO] Loading captions from {caption_json_path}...")
            with open(caption_json_path, 'r', encoding='utf-8') as f:
                self.caption_dict = json.load(f)
            print(f"[INFO] Loaded {len(self.caption_dict)} captions")
        else:
            print(f"[WARN] Caption file not found: {caption_json_path}, using empty captions")
        
        # Load dataset in streaming mode
        print(f"[INFO] Loading dataset: {DATASET_ID} (split={split})...")
        dataset = load_dataset(DATASET_ID, split="train", streaming=True)
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        
        # Convert to list (for indexing) - but limit to max_samples if specified
        self.samples = []
        count = 0
        for sample in dataset:
            if max_samples is not None and count >= max_samples:
                break
            # Filter out invalid samples
            if self._is_valid_sample(sample):
                self.samples.append(sample)
                count += 1
        
        print(f"[INFO] Loaded {len(self.samples)} samples for {split} split")
    
    def _is_valid_sample(self, sample):
        """Check if sample is valid for training."""
        if 'preview' not in sample or sample['preview'] is None:
            return False
        if 'left' not in sample or len(sample['left']) == 0:
            return False
        return True
    
    def _get_caption_from_json(self, img_id) -> str:
        """
        Get caption from caption_llava15.json based on image ID.
        
        The JSON file uses full paths as keys (e.g., /workspace/dataset/.../00000000.png),
        so we try to match by filename (e.g., 00000000.png).
        
        Args:
            img_id: Image ID from the dataset sample (can be int, str, or path)
            
        Returns:
            Caption string, or empty string if not found
        """
        if not self.caption_dict:
            return ""
        
        # Convert img_id to string
        img_id_str = str(img_id)
        
        # Try direct match first (in case img_id is a full path)
        if img_id_str in self.caption_dict:
            return self.caption_dict[img_id_str]
        
        # Normalize img_id to get the numeric part
        # Handle cases like: "0", "00000000", "00000000.png", "/path/to/00000000.png"
        img_id_normalized = img_id_str
        
        # Extract filename if it's a path
        if '/' in img_id_str or '\\' in img_id_str:
            img_id_normalized = Path(img_id_str).stem
        else:
            # Remove extension if present
            img_id_normalized = Path(img_id_str).stem
        
        # Try to convert to int and format as 8-digit zero-padded string
        try:
            img_id_num = int(img_id_normalized)
            img_id_padded = f"{img_id_num:08d}"
        except ValueError:
            img_id_padded = img_id_normalized
        
        # Search through all keys in caption_dict to find matching filename
        # JSON keys are like: /workspace/dataset/.../00000000.png
        for key, caption in self.caption_dict.items():
            key_filename = Path(key).name  # e.g., "00000000.png"
            key_stem = Path(key).stem       # e.g., "00000000"
            
            # Match by full filename
            if key_filename == f"{img_id_padded}.png" or key_filename == img_id_str:
                return caption
            
            # Match by stem (without extension)
            if key_stem == img_id_padded or key_stem == img_id_normalized:
                return caption
        
        return ""
    
    def _generate_depth(self, rgb_image: Image.Image) -> Optional[np.ndarray]:
        """
        Generate depth map for an RGB image using ml-depth-pro.
        
        Args:
            rgb_image: PIL Image in RGB format
            
        Returns:
            Normalized depth map as numpy array (H, W) in range [0, 1], or None if failed
        """
        if not self.use_depth or self.depth_model is None:
            return None
        
        try:
            # Convert PIL Image to numpy array (H, W, 3)
            img_array = np.array(rgb_image)
            
            # Transform image for depth model
            # The transform expects numpy array and returns tensor
            img_tensor = self.depth_transform(img_array)
            
            # Run inference
            with torch.no_grad():
                prediction = self.depth_model.infer(img_tensor.to(self.depth_device), f_px=None)
                depth = prediction["depth"]  # Depth in meters
            
            # Convert to numpy and normalize to [0, 1]
            depth_np = depth.detach().cpu().numpy()
            if depth_np.ndim == 3:
                depth_np = depth_np.squeeze()
            elif depth_np.ndim == 4:
                depth_np = depth_np.squeeze(0).squeeze(0)
            
            # Normalize depth to [0, 1] range
            # Clip to reasonable range (e.g., 0.1m to 100m) for better normalization
            depth_clipped = np.clip(depth_np, 0.1, 100.0)
            depth_normalized = (depth_clipped - 0.1) / (100.0 - 0.1)
            
            # Ensure same size as input image
            if depth_normalized.shape != (rgb_image.height, rgb_image.width):
                try:
                    from scipy import ndimage
                    depth_normalized = ndimage.zoom(
                        depth_normalized,
                        (rgb_image.height / depth_normalized.shape[0], rgb_image.width / depth_normalized.shape[1]),
                        order=1
                    )
                except ImportError:
                    # Fallback to PIL resize if scipy not available
                    depth_pil = Image.fromarray((depth_normalized * 255).astype(np.uint8))
                    depth_pil = depth_pil.resize((rgb_image.width, rgb_image.height), Image.BILINEAR)
                    depth_normalized = np.array(depth_pil).astype(np.float32) / 255.0
            
            return depth_normalized.astype(np.float32)
        except Exception as e:
            print(f"[WARN] Failed to generate depth: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load and process a single sample."""
        sample = self.samples[idx]
        
        # Get the main image
        main_img = sample['preview'].convert("RGB")
        W, H = main_img.size
        
        # Generate depth map for the whole image if enabled
        depth_map = None
        if self.use_depth:
            depth_map = self._generate_depth(main_img)
        
        # Get canvas size
        canvas_w = float(sample.get('canvas_width', W))
        canvas_h = float(sample.get('canvas_height', H))
        
        # Get layer information
        l_left = sample.get('left', [])
        l_top = sample.get('top', [])
        l_width = sample.get('width', [])
        l_height = sample.get('height', [])
        l_angle = sample.get('angle', [])
        l_imgs = sample.get('image', [])
        
        # Prepare layer images and boxes
        layer_image_RGBA = []
        layer_image_RGB = []
        layout = []
        
        # Helper function to add depth channel to image
        def add_depth_channel(img_rgb, depth_map_local):
            """Add depth channel to RGB image, creating RGBD format."""
            if depth_map_local is None:
                return img_rgb.convert("RGBA")  # No depth, use RGBA
            
            # Convert RGB to numpy
            img_array = np.array(img_rgb)
            
            # Normalize depth map to [0, 255] for uint8
            depth_uint8 = (depth_map_local * 255).astype(np.uint8)
            
            # Stack RGB + Depth to create RGBD
            img_rgbd = np.dstack([img_array, depth_uint8])
            
            # Convert back to PIL Image (RGBA format, but D channel is depth)
            return Image.fromarray(img_rgbd, mode="RGBA")
        
        # First layer: whole image
        main_img_RGBA = main_img.convert("RGBA")
        if depth_map is not None:
            # Add depth channel to create RGBD
            main_img_RGBD = add_depth_channel(main_img, depth_map)
            layer_image_RGBA.append(self.to_tensor(main_img_RGBD))
        else:
            layer_image_RGBA.append(self.to_tensor(main_img_RGBA))
        layer_image_RGB.append(self.to_tensor(main_img))
        layout.append([0, 0, W - 1, H - 1])
        
        # Second layer: base image (background) - same as whole image for this dataset
        # In the original CLD dataset, base_image is the background layer
        # For DLCV dataset, we use the whole image as base since we don't have separate background
        if depth_map is not None:
            base_img_RGBD = add_depth_channel(main_img, depth_map)
            layer_image_RGBA.append(self.to_tensor(base_img_RGBD))
        else:
            layer_image_RGBA.append(self.to_tensor(main_img_RGBA))
        layer_image_RGB.append(self.to_tensor(main_img))
        layout.append([0, 0, W - 1, H - 1])
        
        # Process each layer
        valid_layers = []
        for i in range(len(l_left)):
            meta_left = float(l_left[i])
            meta_top = float(l_top[i])
            meta_w = float(l_width[i])
            meta_h = float(l_height[i])
            meta_angle = float(l_angle[i]) if i < len(l_angle) else 0.0
            
            # Filter out full background layers
            if meta_w * meta_h > (canvas_w * canvas_h * 0.95):
                continue
            
            # Calculate bounding box
            final_box = None
            
            # Case A: Rotation -> Force pure geometry
            if abs(meta_angle) > 1.0:
                min_x, min_y, max_x, max_y = calculate_rotated_aabb(
                    meta_left, meta_top, meta_w, meta_h, meta_angle
                )
                final_box = (min_x, min_y, max_x, max_y)
            # Case B: No rotation -> Try Alpha Crop
            else:
                if i < len(l_imgs) and l_imgs[i] is not None:
                    try:
                        final_box = get_tight_box_with_scale(
                            l_imgs[i], meta_left, meta_top, meta_w, meta_h
                        )
                    except:
                        pass
                
                # If Alpha Crop fails, revert to geometry
                if final_box is None:
                    final_box = (meta_left, meta_top, meta_left + meta_w, meta_top + meta_h)
            
            # Clamp box to canvas bounds
            raw_x1, raw_y1, raw_x2, raw_y2 = final_box
            x1 = max(0, min(int(raw_x1), W - 1))
            y1 = max(0, min(int(raw_y1), H - 1))
            x2 = max(0, min(int(raw_x2), W - 1))
            y2 = max(0, min(int(raw_y2), H - 1))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Create layer canvas
            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))
            
            # Get layer image if available
            if i < len(l_imgs) and l_imgs[i] is not None:
                try:
                    layer_img = l_imgs[i]
                    if not isinstance(layer_img, Image.Image):
                        layer_img = Image.open(layer_img).convert("RGBA")
                    else:
                        layer_img = layer_img.convert("RGBA")
                    
                    # Resize layer to match metadata size
                    layer_w, layer_h = layer_img.size
                    if layer_w != meta_w or layer_h != meta_h:
                        layer_img = layer_img.resize((int(meta_w), int(meta_h)), Image.BILINEAR)
                    
                    # Apply rotation if needed
                    if abs(meta_angle) > 1.0:
                        layer_img = layer_img.rotate(-meta_angle, expand=True, fillcolor=(0, 0, 0, 0))
                        # Recalculate size after rotation
                        rot_w, rot_h = layer_img.size
                        paste_x = int(meta_left - (rot_w - meta_w) / 2)
                        paste_y = int(meta_top - (rot_h - meta_h) / 2)
                    else:
                        paste_x = int(meta_left)
                        paste_y = int(meta_top)
                    
                    # Paste onto canvas
                    canvas_RGBA.paste(layer_img, (paste_x, paste_y), layer_img)
                    layer_RGB = rgba2rgb(layer_img)
                    canvas_RGB.paste(layer_RGB, (paste_x, paste_y))
                except Exception as e:
                    # If layer image processing fails, create empty layer
                    pass
            
            # Add to lists
            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([x1, y1, x2, y2])
            valid_layers.append(i)
        
        # Ensure we have at least base layers (whole image + base image)
        # We already have whole image and base image, so we're good
        
        # Get caption from caption_llava15.json using image ID
        img_id = sample.get('id', '')
        caption = self._get_caption_from_json(img_id)
        
        # Fallback to sample caption if not found in JSON
        if not caption:
            caption = sample.get('caption', '')
        
        return {
            "pixel_RGBA": layer_image_RGBA,
            "pixel_RGB": layer_image_RGB,
            "whole_img": main_img,
            "caption": caption,
            "height": H,
            "width": W,
            "layout": layout,
        }


def prepare_dlcv_cld_dataset_splits(
    train_max_samples: Optional[int] = 20000,
    val_max_samples: Optional[int] = 2000,
    train_seed: int = 42,
    val_seed: int = 43,
):
    """
    Prepare train/val splits for CLD training.
    This is a helper function to create dataset instances with proper splits.
    
    Returns:
        train_dataset, val_dataset
    """
    train_dataset = DLCVCLDDataset(
        split="train",
        seed=train_seed,
        max_samples=train_max_samples,
    )
    
    val_dataset = DLCVCLDDataset(
        split="val",
        seed=val_seed,
        max_samples=val_max_samples,
    )
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    print("Testing DLCVCLDDataset...")
    dataset = DLCVCLDDataset(split="train", max_samples=10)
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Number of layers: {len(sample['pixel_RGBA'])}")
        print(f"Image size: {sample['width']}x{sample['height']}")
        print(f"Layout boxes: {len(sample['layout'])}")
        print(f"Caption: '{sample['caption']}'")
        print("[INFO] Dataset test passed!")

