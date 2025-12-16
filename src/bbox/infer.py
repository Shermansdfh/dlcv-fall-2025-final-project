"""
RT-DETR Preprocessing: Generate RTDETR detection input for `LayerDecompositionPipeline`.

Based on the logic of `inference_rtdetr.py`, wrapped into function interface:
- Load RTDETR weights
- Predict a single image, return boxes and basic image information
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Point to vendored Ultralytics RT-DETR source
RT_DETR_DIR = REPO_ROOT / "third_party" / "ultralytics" / "ultralytics"
if str(RT_DETR_DIR) not in sys.path:
    sys.path.insert(0, str(RT_DETR_DIR))

from ultralytics import RTDETR

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "rtdetr"

# default weight path follows inference_rtdetr.py
MODEL_PATH = str(CHECKPOINT_DIR / "rtdetr_dlcv_bbox_dataset" / "weights" / "best.pt")


@dataclass
class RTDETRRecord:
    """RTDETR output structure for `LayerDecompositionPipeline`."""

    image_path: str
    orig_img: np.ndarray  # BGR
    image_size: Tuple[int, int]  # (H, W)
    boxes: List[List[float]]  # [x1, y1, x2, y2, conf, cls]


def load_rtdetr(model_path: str = MODEL_PATH) -> RTDETR:
    """Load RTDETR model."""
    return RTDETR(model_path)


def predict_single_image(
    model: RTDETR,
    image_path: str,
    conf: float = 0.4,
) -> RTDETRRecord:
    """
    Predict a single image with RTDETR, return fields needed for `LayerDecompositionPipeline`.
    """
    results = model.predict(source=image_path, conf=conf, save=False, verbose=False)
    if len(results) == 0:
        raise ValueError(f"No prediction returned for {image_path}")

    result = results[0]
    boxes: List[List[float]] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf_val = float(box.conf[0])
        cls_val = int(box.cls[0])
        boxes.append([x1, y1, x2, y2, conf_val, cls_val])

    orig = result.orig_img  # BGR numpy array
    h, w = orig.shape[:2]

    return RTDETRRecord(
        image_path=result.path,
        orig_img=orig,
        image_size=(h, w),
        boxes=boxes,
    )


def batch_predict(
    model: RTDETR,
    sources: Sequence[str],
    conf: float = 0.4,
    limit: Optional[int] = None,
) -> List[RTDETRRecord]:
    """
    Predict multiple images with RTDETR. `sources` can be a list of file paths or a list of paths expanded from a folder glob.
    """
    collected: List[RTDETRRecord] = []
    for idx, image_path in enumerate(sources):
        if limit is not None and idx >= limit:
            break
        rec = predict_single_image(model, image_path, conf=conf)
        collected.append(rec)
    return collected


def save_rtdetr_results(records: List[RTDETRRecord], output_dir: str):
    """Save RTDETR results to JSON files."""
    import json
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for rec in records:
        stem = Path(rec.image_path).stem
        result = {
            "image_path": rec.image_path,
            "image_size": list(rec.image_size),
            "boxes": rec.boxes,
        }
        json_path = output_path / f"{stem}.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    
    print(f"✅ Saved {len(records)} RTDETR results to {output_dir}")


def resolve_path(path: str, config_path: Path) -> Path:
    """Resolve relative paths relative to config file location."""
    p = Path(path)
    if p.is_absolute():
        return p
    # Resolve relative to config file's parent directory
    return (config_path.parent / p).resolve()


if __name__ == "__main__":
    import argparse
    import yaml
    from glob import glob
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # RTDETR config - resolve paths relative to config file
    input_dir = resolve_path(config["rtdetr"]["input_dir"], config_path)
    output_dir = resolve_path(config["rtdetr"]["output_dir"], config_path)
    model_path_str = config["rtdetr"].get("model_path", MODEL_PATH)
    model_path = resolve_path(model_path_str, config_path)
    conf = config["rtdetr"].get("conf", 0.4)
    limit = config["rtdetr"].get("limit", None)
    
    # Find images
    image_exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_paths = []
    for ext in image_exts:
        image_paths.extend(input_dir.glob(ext))
        image_paths.extend(input_dir.glob(ext.upper()))
    image_paths = sorted([str(p) for p in image_paths])
    
    if not image_paths:
        print(f"❌ No images found in {input_dir}")
        exit(1)
    
    print(f"🚀 Loading RTDETR model: {model_path}")
    model = load_rtdetr(str(model_path))
    
    print(f"📂 Processing {len(image_paths)} images...")
    records = batch_predict(model, image_paths, conf=conf, limit=limit)
    
    save_rtdetr_results(records, str(output_dir))

