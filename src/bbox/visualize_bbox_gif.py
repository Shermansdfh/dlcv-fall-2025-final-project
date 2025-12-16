#!/usr/bin/env python
"""
From CLD input JSON (containing ordered/quantized boxes) and corresponding image to generate bbox overlay GIF.

Usage:
  # Single file
  python visualize_bbox_gif.py \
    --input /path/to/pipeline_outputs/cld/piccollage_001.json \
    --output /path/to/pipeline_outputs/cld/piccollage_001.gif \
    --use-quantized \
    --duration 500

  # Entire folder（recursively find *.json, gif and json output or specify --output-dir）
  python visualize_bbox_gif.py \
    --input /path/to/pipeline_outputs/cld \
    --output-dir /path/to/pipeline_outputs/cld_gif \
    --use-quantized

Default:
  - Use ordered_bboxes, if --use-quantized is specified, use quantized_boxes.
  - Each frame focuses on the current bbox, previous boxes are displayed in semi-transparent gray.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def load_boxes(meta: dict, use_quantized: bool) -> List[Tuple[int, int, int, int]]:
    key = "quantized_boxes" if use_quantized else "ordered_bboxes"
    boxes = meta.get(key)
    if not boxes:
        raise ValueError(f"No '{key}' found in json.")
    return [tuple(map(int, b)) for b in boxes]


def draw_frame(
    base: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    idx: int,
    palette: List[Tuple[int, int, int]],
    font: ImageFont.ImageFont | None,
) -> Image.Image:
    """Draw a single frame: processed boxes in gray, current box highlighted."""
    # Important: GIF does not support true RGBA; if you store RGBA directly and specify transparency index,
    # it will cause the background to become transparent/like being cut out. Here we always overlay RGBA on RGB base image,
    # then convert back to RGB, ensure the output does not contain alpha.
    base_rgb = base.convert("RGB")
    overlay = Image.new("RGBA", base_rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Processed boxes (gray)
    for j in range(idx):
        x0, y0, x1, y1 = boxes[j]
        x0 = max(0, min(x0, base_rgb.width - 1))
        y0 = max(0, min(y0, base_rgb.height - 1))
        x1 = max(0, min(x1, base_rgb.width - 1))
        y1 = max(0, min(y1, base_rgb.height - 1))
        draw.rectangle([x0, y0, x1, y1], outline=(128, 128, 128, 160), width=3)

    # Current box (color)
    x0, y0, x1, y1 = boxes[idx]
    x0 = max(0, min(x0, base_rgb.width - 1))
    y0 = max(0, min(y0, base_rgb.height - 1))
    x1 = max(0, min(x1, base_rgb.width - 1))
    y1 = max(0, min(y1, base_rgb.height - 1))
    color = palette[idx % len(palette)]
    draw.rectangle([x0, y0, x1, y1], outline=color + (220,), width=4)
    label = f"layer {idx}"
    if font:
        try:
            # Pillow >=10 recommend using getbbox
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            # Some versions of FreeTypeFont don't have getbbox, so we fall back to textbbox
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = draw.textlength(label), 14
    pad = 4
    box_bg = (0, 0, 0, 140)
    # label is placed above the box; if it exceeds the screen, it is placed inside the box
    y_top = y0 - th - pad * 2
    if y_top < 0:
        y_top = y0 + pad
        y_bottom = y_top + th + pad * 2
    else:
        y_bottom = y0
    x_left = x0
    x_right = min(base_rgb.width, x_left + int(tw) + pad * 2)
    draw.rectangle([x_left, y_top, x_right, y_bottom], fill=box_bg)
    draw.text((x_left + pad, y_top + pad), label, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(base_rgb.convert("RGBA"), overlay).convert("RGB")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CLD bboxes as GIF.")
    parser.add_argument("--input", required=True, type=Path, help="Single JSON or folder（recursively find *.json）")
    parser.add_argument("--output", type=Path, help="Output GIF（only used in single file mode）")
    parser.add_argument("--output-dir", type=Path, help="Output directory for folder mode; default is the same level as JSON")
    parser.add_argument("--use-quantized", action="store_true", help="Use quantized_boxes, not ordered_bboxes")
    parser.add_argument("--duration", type=int, default=500, help="Duration of each frame in milliseconds")
    parser.add_argument("--loop", type=int, default=0, help="GIF loop times, 0 means infinite loop")
    args = parser.parse_args()
    args = parser.parse_args()

    input_path: Path = args.input
    targets: list[Path] = []
    if input_path.is_dir():
        targets = list(input_path.rglob("*.json"))
        if not targets:
            raise FileNotFoundError(f"No json found under {input_path}")
    else:
        targets = [input_path]

    # Try to load system font; if failed, use default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = None

    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (241, 144, 102),
        (42, 157, 143),
        (244, 162, 97),
        (233, 196, 106),
    ]

    for json_path in targets:
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        image_path = Path(meta["image_path"])
        if not image_path.exists():
            print(f"[WARN] image_path not found, skip: {image_path}")
            continue

        boxes = load_boxes(meta, use_quantized=args.use_quantized)
        base_img = Image.open(image_path).convert("RGB")

        frames = [draw_frame(base_img, boxes, i, palette, font) for i in range(len(boxes))]

        if args.output and not input_path.is_dir():
            output = args.output
        else:
            out_dir = args.output_dir if args.output_dir else json_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"{json_path.stem}.gif"

        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=args.duration,
            loop=args.loop,
            disposal=2,
            optimize=False,
        )
        print(f"[INFO] Saved GIF to {output}")


if __name__ == "__main__":
    main()

