"""
Turn RTDETR detection boxes + LayerD layer masks into CLD inference format.

Overview:
- Load RTDETR detection boxes (json)
- Load LayerD masks (npz)
- Assign each box to the most matching mask using IoF (Intersection-over-Foreground)
- If a box is assigned to a background mask, trigger "background rescue"
- Output CLD inference required fields:
  - ordered_bboxes (int xyxy)
  - layer_indices (1-based layers; background layer implicitly 0)
  - caption / whole_caption (maintain empty string here, filled in later steps)
  - quantized_boxes (quantized using `tools.tools.get_input_box` with 16x quantization)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

import numpy as np

def get_input_box(
    layer_boxes: Sequence[Sequence[float]], 
    img_width: Optional[int] = None, 
    img_height: Optional[int] = None
) -> List[Tuple[int, int, int, int]]:
    """
    Quantize xyxy boxes to CLD's VAE stride (8px grid).
    
    VAE stride is 8 (from patchify patch_size=8 and pipeline bbox // 8 conversion).
    This ensures bbox coordinates are multiples of 8, which is required for proper
    latent space conversion (bbox coordinates are divided by 8 in pipeline).

    Args:
        layer_boxes: List of [x1, y1, x2, y2] boxes
        img_width: Image width (optional, for boundary clamping)
        img_height: Image height (optional, for boundary clamping)

    - min coords: floor to nearest multiple of 8
    - max coords: ceil to nearest multiple of 8 (preserves already-aligned coordinates)
    """
    list_layer_box: List[Tuple[int, int, int, int]] = []
    for layer_box in layer_boxes:
        min_row, max_row = float(layer_box[1]), float(layer_box[3])
        min_col, max_col = float(layer_box[0]), float(layer_box[2])

        # Floor to nearest multiple of 8 (min coordinates)
        quantized_min_row = (int(min_row) // 8) * 8
        quantized_min_col = (int(min_col) // 8) * 8
        
        # Ceil to nearest multiple of 8 (max coordinates)
        # Use ((max + 7) // 8) * 8 instead of ((max // 8) + 1) * 8
        # This preserves already-aligned coordinates (e.g., 800 stays 800, not 808)
        quantized_max_row = ((int(max_row) + 7) // 8) * 8
        quantized_max_col = ((int(max_col) + 7) // 8) * 8
        
        # Clamp to image boundaries if provided
        if img_width is not None:
            quantized_min_col = max(0, min(quantized_min_col, img_width))
            quantized_max_col = max(quantized_min_col, min(quantized_max_col, img_width))
        if img_height is not None:
            quantized_min_row = max(0, min(quantized_min_row, img_height))
            quantized_max_row = max(quantized_min_row, min(quantized_max_row, img_height))

        list_layer_box.append((quantized_min_col, quantized_min_row, quantized_max_col, quantized_max_row))
    return list_layer_box


class LayerMatchResult(TypedDict):
    """Debug record of assigning a single RTDETR box to a LayerD mask."""

    box_index: int
    mask_index: int
    iof: float
    rescued_from_background: bool


class ProcessOutput(TypedDict):
    """Structured output for CLD inference preprocessing."""

    origin_image: Optional[np.ndarray]
    caption: str
    whole_caption: str
    ordered_bboxes: List[List[int]]
    layer_indices: List[int]
    debug_info: Dict[str, List[LayerMatchResult]]


def match_layers(
    boxes: np.ndarray,
    masks: Sequence[np.ndarray],
) -> Tuple[List[int], List[LayerMatchResult]]:
    """
    Assign each RTDETR box to the LayerD mask with the highest IoF.

    IoF = intersection_area / box_area
    「background rescue」不在這裡做，由上游 pipeline 處理。
    """
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError("boxes must have shape [N, >=4]")
    if not masks:
        raise ValueError("masks must be non-empty")

    h, w = masks[0].shape[:2]
    assignments: List[int] = []
    debug: List[LayerMatchResult] = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        x1i, y1i = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        x2i, y2i = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)

        if x2i <= x1i or y2i <= y1i:
            background_idx = len(masks) - 1
            assignments.append(background_idx)  # degenerate box -> background
            debug.append(
                LayerMatchResult(
                    box_index=i,
                    mask_index=background_idx,
                    iof=0.0,
                    rescued_from_background=False,
                )
            )
            continue

        box_mask = np.zeros((h, w), dtype=bool)
        box_mask[y1i:y2i, x1i:x2i] = True
        box_area = float((x2i - x1i) * (y2i - y1i))

        best_idx, best_iof = 0, -1.0
        for m_idx, m in enumerate(masks):
            inter = np.logical_and(box_mask, m > 0)
            iof = float(np.sum(inter)) / max(box_area, 1e-6)
            if iof > best_iof:
                best_iof = iof
                best_idx = m_idx

        assignments.append(best_idx)
        debug.append(
            LayerMatchResult(
                box_index=i,
                mask_index=best_idx,
                iof=best_iof,
                rescued_from_background=False,
            )
        )

    return assignments, debug


@dataclass
class LayerDecompositionPipeline:
    """
    Combine RTDETR detection boxes + LayerD masks into a CLD usable layer ordering.

    Core rules:
    - RTDETR boxes define object presence and geometry.
    - LayerD masks only provide relative depth (layer).
    - Boxes assigned to background will be "rescued" to the deepest foreground layer (maintained by existing logic in this file).
    - Any LayerD mask without any RTDETR support (orphan) will be ignored.
    """

    def process(
        self,
        image_size: Tuple[int, int],
        RTDETR_boxes: Sequence[Sequence[float]],
        layerd_masks: Sequence[np.ndarray],
        caption: str,
    ) -> ProcessOutput:
        """
        Args:
            image_size: (height, width) same as the image size used for RTDETR inference.
            RTDETR_boxes: [[x1, y1, x2, y2, confidence, class_id], ...]
            layerd_masks: masks list (the last one is usually background).
            caption: image caption (this process will return it as is).
        """
        height, width = image_size
        boxes = np.asarray(RTDETR_boxes, dtype=float)
        whole_caption = caption

        if boxes.size == 0:
            # Even with no boxes, add a background bbox
            background_bbox = [0, 0, width - 1, height - 1]
            background_debug = LayerMatchResult(
                box_index=-1,
                mask_index=-1,
                iof=1.0,
                rescued_from_background=False,
            )
            return ProcessOutput(
                origin_image=None,
                caption=caption,
                whole_caption=whole_caption,
                ordered_bboxes=[background_bbox],
                layer_indices=[0],  # Background layer
                debug_info={"matches": [background_debug]},
            )

        masks = [self._ensure_mask_size(m, height, width) for m in layerd_masks]
        assignments, debug_matches = match_layers(boxes, masks)

        background_idx = len(masks) - 1
        rescued_assignments: List[int] = []
        for idx, mask_idx in enumerate(assignments):
            if mask_idx == background_idx:
                # Background rescue: maintain existing behavior (shift to layer index 1).
                assignments[idx] = 0  # temp marker, will become layer 1 after shift
                debug_matches[idx]["rescued_from_background"] = True
            rescued_assignments.append(assignments[idx])

        # Convert to 1-based layer indices (background layer implicitly 0).
        layer_indices = [mask_idx + 1 for mask_idx in rescued_assignments]

        # Sort by layer depth from smallest to largest; larger areas are considered deeper (maintain original sorting rules).
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = sorted(range(len(boxes)), key=lambda i: (layer_indices[i], -areas[i]))

        ordered_bboxes = [[int(round(c)) for c in boxes[i, :4]] for i in order]
        ordered_layers = [int(layer_indices[i]) for i in order]
        ordered_debug = [debug_matches[i] for i in order]

        # Always add a full-image bbox as background layer (layer index 0)
        # This ensures CLD inference always has a background layer
        background_bbox = [0, 0, width - 1, height - 1]
        ordered_bboxes.insert(0, background_bbox)
        ordered_layers.insert(0, 0)  # Background layer index is 0
        # Add a debug entry for the background bbox
        background_debug = LayerMatchResult(
            box_index=-1,  # Special index for background bbox
            mask_index=-1,
            iof=1.0,  # Full image coverage
            rescued_from_background=False,
        )
        ordered_debug.insert(0, background_debug)

        return ProcessOutput(
            origin_image=None,
            caption=caption,
            whole_caption=whole_caption,
            ordered_bboxes=ordered_bboxes,
            layer_indices=ordered_layers,
            debug_info={"matches": ordered_debug},
        )

    @staticmethod
    def _ensure_mask_size(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        """Ensure mask size matches image; if not, use NEAREST resize."""
        if mask.shape[0] == height and mask.shape[1] == width:
            return mask.astype(float)

        try:
            from PIL import Image  # pillow only imported when resizing is needed
        except Exception as exc:  # pragma: no cover - import guard
            raise ValueError("Mask size mismatch and Pillow is unavailable for resizing.") from exc

        mode = "F" if mask.dtype.kind == "f" else "L"
        pil_img = Image.fromarray(mask, mode=mode)
        resized = pil_img.resize((width, height), resample=Image.NEAREST)
        return np.asarray(resized, dtype=float)


@dataclass
class CLDSample:
    """
    Final data structure for CLD inference.
    """

    image_path: str
    ordered_bboxes: List[List[int]]          # original ordered boxes, quantized
    quantized_boxes: List[tuple[int, int, int, int]]  # after get_input_box quantized
    layer_indices: List[int]                 # 1-based foreground layer indices
    caption: str                             # original caption
    whole_caption: str                       # Caption (generated in Step 3.5 or empty)
    debug_info: dict
    origin_image: Optional[object] = None    # optional, preserve original image


def load_rtdetr_results(rtdetr_output_dir: str) -> dict:
    """Load RTDETR results from JSON files."""
    import json

    results = {}
    rtdetr_path = Path(rtdetr_output_dir)
    for json_file in sorted(rtdetr_path.glob("*.json")):
        stem = json_file.stem
        data = json.loads(json_file.read_text(encoding="utf-8"))
        results[stem] = data
    return results


def load_layerd_results(layerd_output_dir: str) -> dict:
    """Load LayerD masks from NPZ files."""
    results = {}
    layerd_path = Path(layerd_output_dir)
    for npz_file in sorted(layerd_path.glob("*.npz")):
        stem = npz_file.stem
        with np.load(npz_file, allow_pickle=True) as data:
            masks_data = data["masks"]

            # Handle both array and list formats
            if isinstance(masks_data, np.ndarray):
                masks_list = [masks_data[i] for i in range(len(masks_data))]
            else:
                masks_list = list(masks_data)

            results[stem] = {
                "masks": masks_list,
                "image_size": tuple(data["image_size"]),
                "image_path": str(data["image_path"]),
            }
    return results


def pipeline_to_cld_samples(
    rtdetr_output_dir: str,
    layerd_output_dir: str,
) -> Iterable[CLDSample]:
    """
    Load RTDETR and LayerD results, generate CLD inference samples.
    """
    rtdetr_results = load_rtdetr_results(rtdetr_output_dir)
    layerd_results = load_layerd_results(layerd_output_dir)
    pipeline = LayerDecompositionPipeline()

    # Match by stem
    for stem, rtdetr_data in rtdetr_results.items():
        if stem not in layerd_results:
            print(f"⚠️  Skipping {stem}: LayerD result not found")
            continue
        
        layerd_data = layerd_results[stem]
        
        # Process with pipeline
        po: ProcessOutput = pipeline.process(
            image_size=tuple(rtdetr_data["image_size"]),
            RTDETR_boxes=rtdetr_data["boxes"],
            layerd_masks=layerd_data["masks"],
            caption="",  # Empty caption, will be filled in Step 3.5
        )

        # Get image dimensions for boundary clamping
        height, width = rtdetr_data["image_size"]
        quantized = get_input_box(po["ordered_bboxes"], img_width=width, img_height=height)

        yield CLDSample(
            image_path=rtdetr_data["image_path"],
            ordered_bboxes=po["ordered_bboxes"],
            quantized_boxes=quantized,
            layer_indices=po["layer_indices"],
            caption=po["caption"],
            whole_caption=po["whole_caption"],
            debug_info=po["debug_info"],
            origin_image=po["origin_image"],
        )


def save_cld_samples(samples: Iterable[CLDSample], output_dir: str) -> None:
    """Save CLD samples to JSON files."""
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for sample in samples:
        stem = Path(sample.image_path).stem
        # Convert image_path to absolute path for reliable loading in CLD inference
        image_path_abs = Path(sample.image_path).resolve()
        if not image_path_abs.exists():
            # If absolute path doesn't exist, try to find it relative to output_dir
            # This handles cases where image_path was relative
            potential_path = output_path.parent.parent / sample.image_path
            if potential_path.exists():
                image_path_abs = potential_path.resolve()
            else:
                print(f"⚠️  Warning: Image path may not be accessible: {sample.image_path}")

        result = {
            "image_path": str(image_path_abs),  # Always use absolute path
            "ordered_bboxes": sample.ordered_bboxes,
            "quantized_boxes": [list(b) for b in sample.quantized_boxes],
            "layer_indices": sample.layer_indices,
            "caption": sample.caption,
            "whole_caption": sample.whole_caption,
            "debug_info": sample.debug_info,
        }
        json_path = output_path / f"{stem}.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        count += 1

    print(f"✅ Saved {count} CLD samples to {output_dir}")


def resolve_path(path: str, config_path: Path) -> str:
    """Resolve relative paths relative to config file location."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    # Resolve relative to config file's parent directory
    return str((config_path.parent / p).resolve())


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # CLD config - resolve paths relative to config file
    rtdetr_output_dir = resolve_path(config["cld"]["rtdetr_output_dir"], config_path)
    layerd_output_dir = resolve_path(config["cld"]["layerd_output_dir"], config_path)
    output_dir = resolve_path(config["cld"]["output_dir"], config_path)

    print(f"📂 Loading RTDETR results from {rtdetr_output_dir}")
    print(f"📂 Loading LayerD results from {layerd_output_dir}")

    samples = pipeline_to_cld_samples(
        rtdetr_output_dir=rtdetr_output_dir,
        layerd_output_dir=layerd_output_dir,
    )

    save_cld_samples(samples, output_dir)


__all__ = [
    "CLDSample",
    "pipeline_to_cld_samples",
    "load_rtdetr_results",
    "load_layerd_results",
    "save_cld_samples",
]


