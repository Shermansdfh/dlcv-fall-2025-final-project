#!/usr/bin/env python
"""
Step 4: CLD Inference

Calls src/cld/infer_dlcv.py using conda CLD environment.
This step requires a separate CLD inference config file (e.g., configs/exp001/cld/infer.yaml).
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


def run_step4_cld(
    pipeline_config_path: Path,
    cld_infer_config_path: Path = None,
    conda_env: str = None
) -> int:
    """
    Run CLD inference step.
    
    Args:
        pipeline_config_path: Path to pipeline.yaml config file
        cld_infer_config_path: Path to CLD inference config (e.g., configs/exp001/cld/infer.yaml)
                              If None, will try to infer from pipeline config location
        conda_env: Conda environment name (default: read from config or "CLD")
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import yaml
    
    # Load config to get conda env
    with open(pipeline_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Get conda env from config if not provided
    if conda_env is None:
        conda_env = config.get("cld_conda_env", config.get("conda_env", "CLD"))
    
    # Determine CLD inference config path
    if cld_infer_config_path is None:
        # Try to infer from pipeline config location
        # e.g., configs/exp001/pipeline.yaml -> configs/exp001/cld/infer.yaml
        pipeline_config_dir = pipeline_config_path.parent
        cld_infer_config_path = pipeline_config_dir / "cld" / "infer.yaml"
        
        if not cld_infer_config_path.exists():
            # Fallback: try default location
            cld_infer_config_path = REPO_ROOT / "configs" / "exp001" / "cld" / "infer.yaml"
    
    cld_infer_config_path = Path(cld_infer_config_path).resolve()
    
    if not cld_infer_config_path.exists():
        print(f"❌ CLD inference config not found: {cld_infer_config_path}")
        print(f"   Please create it or specify with --cld-infer-config")
        return 1
    
    # Get CLD inference script path
    cld_script = REPO_ROOT / "src" / "cld" / "infer_dlcv.py"
    if not cld_script.exists():
        print(f"❌ CLD inference script not found: {cld_script}")
        return 1
    
    # Build command: conda run -n <env> python <script> --config_path <cld_infer_config>
    cmd = [
        "conda", "run",
        "-n", conda_env,
        "--no-capture-output",  # Show output in real-time
        "python", str(cld_script),
        "--config_path", str(cld_infer_config_path)
    ]
    
    print("=" * 60)
    print("STEP 4: CLD Inference")
    print("=" * 60)
    print(f"🔧 Running: {' '.join(cmd)}")
    print(f"   Environment: {conda_env}")
    print(f"   Pipeline config: {pipeline_config_path}")
    print(f"   CLD inference config: {cld_infer_config_path}")
    print()
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        print("\n✅ Step 4 (CLD Inference) completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step 4 (CLD Inference) failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n❌ Conda not found. Please ensure conda is installed and in PATH")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: CLD Inference")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to pipeline.yaml config file"
    )
    parser.add_argument(
        "--cld-infer-config",
        type=str,
        default=None,
        help="Path to CLD inference config (default: inferred from pipeline config location)"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="CLD",
        help="Conda environment name (default: CLD)"
    )
    args = parser.parse_args()
    
    pipeline_config_path = Path(args.config).resolve()
    if not pipeline_config_path.exists():
        print(f"❌ Pipeline config file not found: {pipeline_config_path}")
        sys.exit(1)
    
    cld_infer_config_path = Path(args.cld_infer_config).resolve() if args.cld_infer_config else None
    exit_code = run_step4_cld(
        pipeline_config_path,
        cld_infer_config_path=cld_infer_config_path,
        conda_env=args.conda_env
    )
    sys.exit(exit_code)

