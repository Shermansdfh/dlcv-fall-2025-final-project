#!/usr/bin/env python3
"""
Post-process CLD inference outputs into a flat ZIP with piccollage-style names.

The output ZIP structure:

your_zip_file_name.zip
├── piccollage_001_0.png  (background_rgba.png from case_0)
├── piccollage_001_1.png  (layer_0_rgba.png from case_0)
├── piccollage_001_2.png  (layer_1_rgba.png from case_0)
├── piccollage_002_0.png  (background_rgba.png from case_1)
├── piccollage_002_1.png  (layer_0_rgba.png from case_1)
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
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from typing import List, Tuple

import yaml


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


def _collect_renamed_outputs(
    inference_dir: Path,
) -> List[Tuple[Path, str]]:
    """
    Collect all layer files from case_* subdirectories and rename them.
    
    Args:
        inference_dir: Directory containing case_* subdirectories
        
    Returns:
        List[(source_path, arcname)]
        - source_path: actual file location
        - arcname: file name in ZIP (e.g., piccollage_001_0.png)
    """
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")
    
    # Find all case_* directories
    case_dirs = sorted(
        [d for d in inference_dir.iterdir() if d.is_dir() and d.name.startswith("case_")],
        key=_extract_case_number
    )
    
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found in {inference_dir}")
    
    results: List[Tuple[Path, str]] = []
    
    for case_dir in case_dirs:
        case_num = _extract_case_number(case_dir)
        # case_n -> piccollage_00(n+1)
        piccollage_num = case_num + 1
        piccollage_prefix = f"piccollage_{piccollage_num:03d}"
        
        # Collect all files from this case
        case_files = _collect_case_files(case_dir)
        
        for file_path, layer_idx in case_files:
            target_name = f"{piccollage_prefix}_{layer_idx}.png"
            results.append((file_path, target_name))
    
    return results


def _create_zip(
    items: List[Tuple[Path, str]],
    zip_path: Path,
) -> None:
    """Create ZIP file and write all items with given arcnames."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in items:
            zf.write(src, arcname=arcname)


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
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = _resolve_repo_root(script_path.parent)

    # Determine input directory
    if args.input_dir:
        inference_dir = Path(args.input_dir)
        if not inference_dir.is_absolute():
            inference_dir = (Path.cwd() / inference_dir).resolve()
    else:
        # Fallback to config file
        config_path = Path(args.config_path)
        if not config_path.is_absolute():
            config_path = (repo_root / config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        cfg = _load_cld_config(config_path)
        save_dir = Path(cfg["save_dir"])
        if not save_dir.is_absolute():
            save_dir = (repo_root / save_dir).resolve()
        inference_dir = save_dir

    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    items = _collect_renamed_outputs(inference_dir=inference_dir)

    output_zip = Path(args.output_zip)
    if not output_zip.is_absolute():
        output_zip = (Path.cwd() / output_zip).resolve()

    print(f"[INFO] Input directory: {inference_dir}")
    print(f"[INFO] Output ZIP: {output_zip}")
    print(f"[INFO] Total files to pack: {len(items)}")

    _create_zip(items, output_zip)

    print(f"[INFO] ZIP written to: {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


