#!/usr/bin/env python
"""
Step 3.5: VLM Caption Generation

Calls src/caption/generate.py using conda llava environment.
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


def run_step3_5_vlm(config_path: Path, conda_env: str = None, force: bool = False) -> int:
    """
    Run VLM caption generation step.
    
    Args:
        config_path: Path to pipeline.yaml config file
        conda_env: Conda environment name (default: read from config or "llava15")
        force: Force regenerate captions even if they already exist
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import yaml
    
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Get conda env from config if not provided
    if conda_env is None:
        conda_env = config.get("vlm_conda_env", "llava15")
    
    # Check if VLM caption is enabled
    step3_5_config = config.get("step3_5", {})
    vlm_config = step3_5_config.get("vlm", {})
    if not vlm_config.get("use_vlm_caption", False):
        print("ℹ️  VLM caption generation disabled (use_vlm_caption=false)")
        return 0
    
    # Get VLM script path
    vlm_script = REPO_ROOT / "src" / "caption" / "generate.py"
    if not vlm_script.exists():
        print(f"❌ VLM script not found: {vlm_script}")
        return 1
    
    # Build command: conda run -n <env> python <script> --config <config> [--force]
    cmd = [
        "conda", "run",
        "-n", conda_env,
        "--no-capture-output",  # Show output in real-time
        "python", str(vlm_script),
        "--config", str(config_path.resolve())
    ]
    
    if force:
        cmd.append("--force")
    
    print("=" * 60)
    print("STEP 3.5: VLM Caption Generation")
    print("=" * 60)
    print(f"🔧 Running: {' '.join(cmd)}")
    print(f"   Environment: {conda_env}")
    print(f"   Config: {config_path}")
    if force:
        print(f"   Force regenerate: True")
    print()
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        print("\n✅ Step 3.5 (VLM Caption) completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step 3.5 (VLM Caption) failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n❌ Conda not found. Please ensure conda is installed and in PATH")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3.5: VLM Caption Generation")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to pipeline.yaml config file"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="llava15",
        help="Conda environment name (default: llava15)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force regenerate captions even if they already exist"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    exit_code = run_step3_5_vlm(config_path, conda_env=args.conda_env, force=args.force)
    sys.exit(exit_code)

