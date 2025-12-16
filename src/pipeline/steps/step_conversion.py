#!/usr/bin/env python
"""
Step 3: CLD Format Conversion

Calls src/adapters/rtdetr_layerd_to_cld_infer.py to convert RTDETR + LayerD results
into CLD inference format.

This step uses conda CLD environment (same as step 4) since it may share dependencies.
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


def run_step3_conversion(config_path: Path, conda_env: str = None) -> int:
    """
    Run CLD format conversion step.
    
    Args:
        config_path: Path to pipeline.yaml config file
        conda_env: Conda environment name (default: read from config or "CLD")
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import yaml
    
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Get conda env from config if not provided
    if conda_env is None:
        conda_env = config.get("cld_conda_env", config.get("conda_env", "CLD"))
    
    # Get conversion script path
    conversion_script = REPO_ROOT / "src" / "adapters" / "rtdetr_layerd_to_cld_infer.py"
    if not conversion_script.exists():
        print(f"❌ Conversion script not found: {conversion_script}")
        return 1
    
    # Build command: conda run -n <env> python <script> --config <config>
    cmd = [
        "conda", "run",
        "-n", conda_env,
        "--no-capture-output",  # Show output in real-time
        "python", str(conversion_script),
        "--config", str(config_path.resolve())
    ]
    
    print("=" * 60)
    print("STEP 3: CLD Format Conversion")
    print("=" * 60)
    print(f"🔧 Running: {' '.join(cmd)}")
    print(f"   Environment: {conda_env}")
    print(f"   Config: {config_path}")
    print()
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        print("\n✅ Step 3 (CLD Format Conversion) completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step 3 (CLD Format Conversion) failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n❌ Conda not found. Please ensure conda is installed and in PATH")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: CLD Format Conversion")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to pipeline.yaml config file"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="CLD",
        help="Conda environment name (default: CLD)"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    exit_code = run_step3_conversion(config_path, conda_env=args.conda_env)
    sys.exit(exit_code)

