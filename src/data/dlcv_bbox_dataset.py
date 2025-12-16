import os
import math
import random
import shutil
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "data" / "dlcv_bbox_dataset"

DATASET_ID = "WalkerHsu/DLCV2025_final_project_piccollage"
PADDING_RATIO = 0.05 

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

def prepare_dlcv_bbox_dataset(target_total=20000, output_dir=DATASET_PATH):
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    print(f"📡 Load Dataset: {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=2000)
    total_count = 0
    
    # Create data.yaml
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\nval: images/val\nnames:\n  0: layout_element")

    print(f"🚀 Start processing (geometry first strategy)...")
    
    for sample in tqdm(dataset, total=target_total):
        if total_count >= target_total: break
        subset = "train" if random.random() < 0.9 else "val"
        
        try:
            if 'preview' not in sample or sample['preview'] is None: continue
            main_img = sample['preview'].convert("RGB")
            img_id = sample['id']
            
            # Get the canvas size (denominator)
            canvas_w = float(sample.get('canvas_width', main_img.size[0]))
            canvas_h = float(sample.get('canvas_height', main_img.size[1]))

            # Get the layer list
            l_left = sample.get('left', [])
            l_top = sample.get('top', [])
            l_width = sample.get('width', [])
            l_height = sample.get('height', [])
            l_angle = sample.get('angle', [])
            l_imgs = sample.get('image', [])
            
            label_lines = []
            has_valid_box = False

            for i in range(len(l_left)):
                meta_left = float(l_left[i])
                meta_top = float(l_top[i])
                meta_w = float(l_width[i])
                meta_h = float(l_height[i])
                # Ensure the angle exists
                meta_angle = float(l_angle[i]) if i < len(l_angle) else 0.0
                
                # Filter the full background
                if meta_w * meta_h > (canvas_w * canvas_h * 0.95): continue
                
                final_box = None # (x1, y1, x2, y2)

                # ==========================================
                # 🚦 Bifurcation logic (The Bifurcation Logic)
                # ==========================================
                
                # Case A: Rotation -> Force pure geometry (Metadata Geometry)
                if abs(meta_angle) > 1.0: # Allow 1 degree of error
                    min_x, min_y, max_x, max_y = calculate_rotated_aabb(
                        meta_left, meta_top, meta_w, meta_h, meta_angle
                    )
                    final_box = (min_x, min_y, max_x, max_y)

                # Case B: No rotation -> Try Alpha Crop (solve cutting text)
                else:
                    if i < len(l_imgs) and l_imgs[i] is not None:
                        try:
                            final_box = get_tight_box_with_scale(
                                l_imgs[i], meta_left, meta_top, meta_w, meta_h
                            )
                        except: pass
                    
                    # If Alpha Crop fails, revert to no rotation geometry
                    if final_box is None:
                        final_box = (meta_left, meta_top, meta_left + meta_w, meta_top + meta_h)

                # ==========================================
                
                # Padding & Clipping
                raw_x1, raw_y1, raw_x2, raw_y2 = final_box
                w_box = raw_x2 - raw_x1
                h_box = raw_y2 - raw_y1
                
                pad_w = w_box * PADDING_RATIO
                pad_h = h_box * PADDING_RATIO
                
                fx1 = max(0, min(raw_x1 - pad_w, canvas_w))
                fy1 = max(0, min(raw_y1 - pad_h, canvas_h))
                fx2 = max(0, min(raw_x2 + pad_w, canvas_w))
                fy2 = max(0, min(raw_y2 + pad_h, canvas_h))
                
                fw, fh = fx2 - fx1, fy2 - fy1

                # Convert YOLO to normalized
                if fw > 2 and fh > 2:
                    cx = (fx1 + fw/2) / canvas_w
                    cy = (fy1 + fh/2) / canvas_h
                    nw = fw / canvas_w
                    nh = fh / canvas_h
                    
                    if 0 <= cx <= 1 and 0 <= cy <= 1:
                        label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        has_valid_box = True

            if has_valid_box:
                save_size = 640
                main_img.resize((save_size, save_size)).save(os.path.join(output_dir, "images", subset, f"{img_id}.jpg"), quality=85)
                with open(os.path.join(output_dir, "labels", subset, f"{img_id}.txt"), "w") as f:
                    f.write("\n".join(label_lines))
                total_count += 1
                
        except Exception: continue

    print(f"\n✅ DLCV Bounding Box Dataset preparation completed! Path: {output_dir}")

if __name__ == "__main__":
    prepare_dlcv_bbox_dataset(target_total=20000)