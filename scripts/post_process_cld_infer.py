#!/usr/bin/env python3
"""
Post-process CLD inference outputs into a flat ZIP with piccollage-style names.

The output ZIP structure:

your_zip_file_name.zip
├── piccollage_001_0.png  (background_rgba.png from case_0, resized to original piccollage_001 dimensions)
├── piccollage_001_1.png  (layer_0_rgba.png from case_0, resized to original piccollage_001 dimensions)
├── piccollage_001_2.png  (layer_1_rgba.png from case_0, resized to original piccollage_001 dimensions)
├── piccollage_002_0.png  (background_rgba.png from case_1, resized to original piccollage_002 dimensions)
├── piccollage_002_1.png  (layer_0_rgba.png from case_1, resized to original piccollage_002 dimensions)
├── ...
└── piccollage_064_N.png

Behavior:
- Process all case_* subfolders in the given directory
- Rename files from each case:
  - case_0: background_rgba.png -> piccollage_001_0.png
  - case_0: layer_0_rgba.png -> piccollage_001_1.png
  - case_0: layer_1_rgba.png -> piccollage_001_2.png
  - case_n: files -> piccollage_00(n+1)_*.png
- All files are placed in a flat ZIP structure
- Images are automatically resized to match the original input image dimensions:
  - piccollage_001_*.png files are resized to original piccollage_001.png WxH
  - piccollage_002_*.png files are resized to original piccollage_002.png WxH
  - And so on...
- Original image dimensions are read from CLD JSON files (e.g., outputs/pipeline_outputs/cld/*.json)
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image


def _load_cld_config(config_path: Path) -> dict:
    """Load CLD infer config YAML."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    return cfg


def _resolve_repo_root(start: Path) -> Path:
    """
    Best-effort repo_root inference, refer to infer_dlcv.py logic:
    - Look up third_party/cld/infer/infer.py
    - If not found, use parents[2] as repo_root
    """
    for p in [start, *start.parents]:
        if (p / "third_party" / "cld" / "infer" / "infer.py").exists():
            return p
    return start.parents[2]


def _extract_case_number(case_dir: Path) -> int:
    """Extract case number from case directory name (e.g., 'case_0' -> 0)."""
    match = re.match(r"case_(\d+)$", case_dir.name)
    if not match:
        raise ValueError(f"Invalid case directory name: {case_dir.name}")
    return int(match.group(1))


def _get_layer_number(layer_file: Path) -> int:
    """Extract layer number from layer file name (e.g., 'layer_0_rgba.png' -> 0)."""
    match = re.match(r"layer_(\d+)_rgba\.png$", layer_file.name)
    if not match:
        raise ValueError(f"Invalid layer file name: {layer_file.name}")
    return int(match.group(1))


def _collect_case_files(case_dir: Path) -> List[Tuple[Path, int]]:
    """
    Collect all layer files from a case directory, sorted by layer index.
    
    Returns:
        List[(file_path, layer_index)]
        - file_path: path to the RGBA file
        - layer_index: 0 for background, 1+ for layers (0-indexed for naming)
    """
    files: List[Tuple[Path, int]] = []
    
    # Add background_rgba.png as layer 0
    background_file = case_dir / "background_rgba.png"
    if background_file.exists():
        files.append((background_file, 0))
    else:
        raise FileNotFoundError(f"background_rgba.png not found in {case_dir}")
    
    # Collect all layer_*_rgba.png files
    layer_files = sorted(case_dir.glob("layer_*_rgba.png"), key=_get_layer_number)
    for layer_file in layer_files:
        layer_idx = _get_layer_number(layer_file)
        # layer_index for naming: background=0, layer_0=1, layer_1=2, etc.
        files.append((layer_file, layer_idx + 1))
    
    return files


def _load_original_image_sizes(
    cld_json_dir: Path,
    repo_root: Path,
) -> Dict[int, Tuple[int, int]]:
    """
    Load original image sizes from CLD JSON files.
    
    Args:
        cld_json_dir: Directory containing CLD JSON files (e.g., outputs/pipeline_outputs/cld)
        repo_root: Repository root path
        
    Returns:
        Dict mapping piccollage number (1-based) to (width, height)
    """
    if not cld_json_dir.exists():
        raise FileNotFoundError(f"CLD JSON directory not found: {cld_json_dir}")
    
    sizes: Dict[int, Tuple[int, int]] = {}
    
    # Find all piccollage_*.json files
    json_files = sorted(cld_json_dir.glob("piccollage_*.json"))
    
    for json_file in json_files:
        # Extract piccollage number from filename (e.g., piccollage_001.json -> 1)
        match = re.match(r"piccollage_(\d+)\.json$", json_file.name)
        if not match:
            continue
        
        piccollage_num = int(match.group(1))
        
        # Load JSON to get image_path
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        image_path = Path(data.get("image_path", ""))
        if not image_path:
            print(f"[WARN] No image_path in {json_file.name}, skipping")
            continue
        
        # Resolve image path (may be absolute or relative)
        if not image_path.is_absolute():
            # Try relative to repo_root
            image_path = (repo_root / image_path).resolve()
        
        if not image_path.exists():
            print(f"[WARN] Original image not found: {image_path}, skipping")
            continue
        
        # Load image to get size
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                sizes[piccollage_num] = (width, height)
        except Exception as e:
            print(f"[WARN] Failed to load image {image_path}: {e}, skipping")
            continue
    
    return sizes


def _collect_renamed_outputs(
    inference_dir: Path,
    cld_json_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> List[Tuple[Path, str, Optional[Tuple[int, int]]]]:
    """
    Collect all layer files from case_* subdirectories and rename them.
    
    Args:
        inference_dir: Directory containing case_* subdirectories
        cld_json_dir: Optional directory containing CLD JSON files for original image sizes
        repo_root: Optional repository root path
        
    Returns:
        List[(source_path, arcname, target_size)]
        - source_path: actual file location
        - arcname: file name in ZIP (e.g., piccollage_001_0.png)
        - target_size: (width, height) to resize to, or None if not available
    """
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")
    
    # Load original image sizes if available
    original_sizes: Dict[int, Tuple[int, int]] = {}
    if cld_json_dir and repo_root:
        try:
            original_sizes = _load_original_image_sizes(cld_json_dir, repo_root)
            print(f"[INFO] Loaded original image sizes for {len(original_sizes)} images")
        except Exception as e:
            print(f"[WARN] Failed to load original image sizes: {e}")
            print(f"[WARN] Images will not be resized")
    
    # Find all case_* directories
    case_dirs = sorted(
        [d for d in inference_dir.iterdir() if d.is_dir() and d.name.startswith("case_")],
        key=_extract_case_number
    )
    
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found in {inference_dir}")
    
    results: List[Tuple[Path, str, Optional[Tuple[int, int]]]] = []
    
    for case_dir in case_dirs:
        case_num = _extract_case_number(case_dir)
        # case_n -> piccollage_00(n+1)
        piccollage_num = case_num + 1
        piccollage_prefix = f"piccollage_{piccollage_num:03d}"
        
        # Get target size for this piccollage number
        target_size = original_sizes.get(piccollage_num)
        
        # Collect all files from this case
        case_files = _collect_case_files(case_dir)
        
        for file_path, layer_idx in case_files:
            target_name = f"{piccollage_prefix}_{layer_idx}.png"
            results.append((file_path, target_name, target_size))
    
    return results


def _resize_image(
    image_path: Path,
    target_size: Tuple[int, int],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Resize image to target size.
    
    Args:
        image_path: Path to source image
        target_size: (width, height) target size
        output_path: Optional output path (if None, creates temp file)
        
    Returns:
        Path to resized image
    """
    if output_path is None:
        # Create temporary file
        output_path = image_path.parent / f"{image_path.stem}_resized{image_path.suffix}"
    
    with Image.open(image_path) as img:
        resized = img.resize(target_size, Image.Resampling.LANCZOS)
        resized.save(output_path, "PNG")
    
    return output_path


def _create_zip(
    items: List[Tuple[Path, str, Optional[Tuple[int, int]]]],
    zip_path: Path,
) -> None:
    """
    Create ZIP file and write all items with given arcnames.
    Resizes images to target size if specified.
    """
    import tempfile
    
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_files: List[Path] = []
    
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src, arcname, target_size in items:
                if target_size is not None:
                    # Resize image before adding to ZIP
                    temp_file = _resize_image(src, target_size)
                    temp_files.append(temp_file)
                    zf.write(temp_file, arcname=arcname)
                else:
                    # No resize needed, add original
                    zf.write(src, arcname=arcname)
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process CLD inference outputs into a flat ZIP with "
            "piccollage-style names (e.g., piccollage_001_0.png)."
        )
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=None,
        help=(
            "Input directory containing case_* subdirectories "
            "(e.g., cld_inference_results). "
            "If not specified, will read from config file."
        ),
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="configs/exp001/cld/infer.yaml",
        help=(
            "CLD inference YAML config path (default: configs/exp001/cld/infer.yaml). "
            "Only used if --input_dir is not specified."
        ),
    )
    parser.add_argument(
        "--output_zip",
        "-o",
        type=str,
        default="cld_inference_postprocessed.zip",
        help="Output ZIP file path (default: cld_inference_postprocessed.zip).",
    )
    parser.add_argument(
        "--cld_json_dir",
        type=str,
        default=None,
        help=(
            "Directory containing CLD JSON files (e.g., outputs/pipeline_outputs/cld). "
            "Used to get original image sizes for resizing. "
            "If not specified, will try to infer from config file."
        ),
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = _resolve_repo_root(script_path.parent)

    # Resolve config path
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    
    # Determine input directory
    cfg: Optional[dict] = None
    if args.input_dir:
        inference_dir = Path(args.input_dir)
        if not inference_dir.is_absolute():
            inference_dir = (Path.cwd() / inference_dir).resolve()
    else:
        # Fallback to config file
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        cfg = _load_cld_config(config_path)
        save_dir = Path(cfg["save_dir"])
        if not save_dir.is_absolute():
            save_dir = (repo_root / save_dir).resolve()
        inference_dir = save_dir

    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    # Determine CLD JSON directory for original image sizes
    cld_json_dir: Optional[Path] = None
    if args.cld_json_dir:
        cld_json_dir = Path(args.cld_json_dir)
        if not cld_json_dir.is_absolute():
            cld_json_dir = (Path.cwd() / cld_json_dir).resolve()
    else:
        # Try to infer from config file
        try:
            if cfg is None and config_path.exists():
                cfg = _load_cld_config(config_path)
            
            if cfg:
                data_dir = Path(cfg.get("data_dir", ""))
                if data_dir:
                    if not data_dir.is_absolute():
                        data_dir = (repo_root / data_dir).resolve()
                    if data_dir.exists():
                        cld_json_dir = data_dir
        except Exception as e:
            print(f"[WARN] Could not infer CLD JSON directory: {e}")

    items = _collect_renamed_outputs(
        inference_dir=inference_dir,
        cld_json_dir=cld_json_dir,
        repo_root=repo_root,
    )

    output_zip = Path(args.output_zip)
    if not output_zip.is_absolute():
        output_zip = (Path.cwd() / output_zip).resolve()

    print(f"[INFO] Input directory: {inference_dir}")
    if cld_json_dir:
        print(f"[INFO] CLD JSON directory: {cld_json_dir}")
    print(f"[INFO] Output ZIP: {output_zip}")
    print(f"[INFO] Total files to pack: {len(items)}")
    
    # Count how many will be resized
    resize_count = sum(1 for _, _, size in items if size is not None)
    if resize_count > 0:
        print(f"[INFO] {resize_count} files will be resized to original image dimensions")

    _create_zip(items, output_zip)

    print(f"[INFO] ZIP written to: {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


