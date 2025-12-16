#!/usr/bin/env python
"""
Step 2: LayerD Decomposition

Calls src/layerd/infer.py using native uv environment.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def resolve_path(path: str, config_path: Path) -> Path:
    """Resolve relative paths relative to config file location."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def run_step2_layerd(config_path: Path, layerd_dir: Path = None) -> int:
    """
    Run LayerD decomposition step.
    
    Args:
        config_path: Path to pipeline.yaml config file
        layerd_dir: Path to LayerD directory (default: third_party/layerd)
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import yaml
    
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Get LayerD directory (where uv project is located)
    if layerd_dir is None:
        layerd_dir = REPO_ROOT / "third_party" / "layerd"
    
    if not layerd_dir.exists():
        print(f"❌ LayerD directory not found: {layerd_dir}")
        return 1
    
    # Get LayerD script path (in src, not in third_party)
    layerd_script = REPO_ROOT / "src" / "layerd" / "infer.py"
    if not layerd_script.exists():
        print(f"❌ LayerD script not found: {layerd_script}")
        return 1
    
    # Build command: uv run python <script> --config <config>
    # Note: We run from layerd_dir to ensure uv can find pyproject.toml
    # But the script is in src/layerd/infer.py, so we use absolute path
    cmd = [
        "uv", "run",
        "python", str(layerd_script),
        "--config", str(config_path.resolve())
    ]
    
    print("=" * 60)
    print("STEP 2: LayerD Decomposition")
    print("=" * 60)
    print(f"🔧 Running: {' '.join(cmd)}")
    print(f"   Working directory: {layerd_dir}")
    print(f"   Config: {config_path}")
    print()
    
    # Run command from layerd_dir to ensure uv environment is correct
    try:
        result = subprocess.run(cmd, check=True, cwd=str(layerd_dir))
        print("\n✅ Step 2 (LayerD) completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step 2 (LayerD) failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n❌ uv not found. Please ensure uv is installed and in PATH")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: LayerD Decomposition")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to pipeline.yaml config file"
    )
    parser.add_argument(
        "--layerd-dir",
        type=str,
        default=None,
        help="Path to LayerD directory (default: third_party/layerd)"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    layerd_dir = Path(args.layerd_dir).resolve() if args.layerd_dir else None
    exit_code = run_step2_layerd(config_path, layerd_dir=layerd_dir)
    sys.exit(exit_code)

