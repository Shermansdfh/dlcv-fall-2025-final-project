import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Point to vendored Ultralytics RT-DETR source
RT_DETR_DIR = REPO_ROOT / "third_party" / "ultralytics" / "ultralytics"
if str(RT_DETR_DIR) not in sys.path:
    sys.path.insert(0, str(RT_DETR_DIR))

# All RT-DETR weights + training runs live under checkpoints/rtdetr
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "rtdetr"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hint Ultralytics to cache weights/configs under checkpoints/rtdetr
os.environ.setdefault("YOLOV8_CACHE_DIR", str(CHECKPOINT_DIR))
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(CHECKPOINT_DIR))

from ultralytics import RTDETR

DATASET_PATH = REPO_ROOT / "data" / "dlcv_bbox_dataset"

def train_rtdetr_final():
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️ Warning: No GPU found, RT-DETR will run forever on CPU!")
    
    # Load pre-trained RT-DETR model
    # rtdetr-l.pt (Large) is the sweet spot for accuracy and speed
    # Force weights (and cache) to live under checkpoints/rtdetr
    model = RTDETR(str(CHECKPOINT_DIR / "rtdetr-l.pt"))

    print("🚀 Start RT-DETR training (Layout Analysis specific settings)...")
    
    results = model.train(
        data=DATASET_PATH / "data.yaml",
        
        # === Training strategy ===
        epochs=100,         # 100 epochs is enough for Fine-tuning
        patience=15,        # Slightly stricter, stop if no improvement for 15 epochs (save compute)
        batch=16,           # [Important] If V100 OOM, please降到 8
        imgsz=640,
        device=0,
        workers=8,          # Accelerate data loading
        
        # === Project settings ===
        # Save training runs under checkpoints/rtdetr/<name>
        project=str(CHECKPOINT_DIR),
        name="rtdetr_dlcv_bbox_dataset",
        exist_ok=True,
        
        # === Optimizer and hyperparameters ===
        optimizer='AdamW',  # Transformer 
        lr0=0.0001,         # Fine-tune smaller learning rate
        
        # === Key: Layout enhancement strategy ===
        mosaic=0.0,         # ❌ Close mosaic (to avoid destroying the layout logic)
        mixup=0.0,          # ❌ Close Mixup (to avoid transparency confusion)
        degrees=0.0,        # ❌ No rotation (layout is usually upright)
        
        # ✅ Keep safe enhancements
        scale=0.5,          # Random scaling (0.5 means +/- 50%) -> Make model adapt to different canvas sizes
        fliplr=0.5,         # Left-right flip (Layout is usually symmetric left-right, unless very concerned about text direction)
        hsv_h=0.015,        # Mild color tone variation
        hsv_s=0.7,          # Saturation variation (Layout color variation is large, can be kept)
        hsv_v=0.4,          # Brightness variation
        
        # === System optimization ===
        cache=True,         # Cache images to RAM, accelerate training of 19K images
        amp=True,           # Automatic mixed precision (save memory, must open)
    )
    
    print(f"🎉 Training completed! Model saved at: {results.save_dir}")

if __name__ == '__main__':
    train_rtdetr_final()
