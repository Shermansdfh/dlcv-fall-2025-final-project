#!/usr/bin/env python
"""
Step 1: RTDETR Detection

Calls src/bbox/infer.py using conda ultralytics environment.
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


def run_step1_rtdetr(config_path: Path, conda_env: str = None) -> int:
    """
    Run RTDETR detection step.
    
    Args:
        config_path: Path to pipeline.yaml config file
        conda_env: Conda environment name (default: read from config or "ultralytics")
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import yaml
    
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Get conda env from config if not provided
    if conda_env is None:
        conda_env = config.get("rtdetr_conda_env", "ultralytics")
    
    # Get RTDETR script path
    rtdetr_script = REPO_ROOT / "src" / "bbox" / "infer.py"
    if not rtdetr_script.exists():
        print(f"❌ RTDETR script not found: {rtdetr_script}")
        return 1
    
    # Build command: conda run -n <env> python <script> --config <config>
    cmd = [
        "conda", "run",
        "-n", conda_env,
        "--no-capture-output",  # Show output in real-time
        "python", str(rtdetr_script),
        "--config", str(config_path.resolve())
    ]
    
    print("=" * 60)
    print("STEP 1: RTDETR Detection")
    print("=" * 60)
    print(f"🔧 Running: {' '.join(cmd)}")
    print(f"   Environment: {conda_env}")
    print(f"   Config: {config_path}")
    print()
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        print("\n✅ Step 1 (RTDETR) completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step 1 (RTDETR) failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n❌ Conda not found. Please ensure conda is installed and in PATH")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: RTDETR Detection")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to pipeline.yaml config file"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="ultralytics",
        help="Conda environment name (default: ultralytics)"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    exit_code = run_step1_rtdetr(config_path, conda_env=args.conda_env)
    sys.exit(exit_code)

