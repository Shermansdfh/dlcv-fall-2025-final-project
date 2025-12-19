#!/usr/bin/env python
"""
Step 4: CLD Inference

Calls src/cld/infer_dlcv.py using conda CLD environment.
This step requires a separate CLD inference config file (e.g., configs/exp001/cld/infer.yaml).
"""

import argparse
import os
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
    conda_env: str = None,
    start_from: int = 0
) -> int:
    """
    Run CLD inference step.
    
    Args:
        pipeline_config_path: Path to pipeline.yaml config file
        cld_infer_config_path: Path to CLD inference config (e.g., configs/exp001/cld/infer.yaml)
                              If None, will try to infer from pipeline config location
        conda_env: Conda environment name (default: read from config or "CLD")
        start_from: Start processing from the Nth JSON file (0-indexed). Useful for resuming interrupted runs.
    
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
    
    # Check CUDA availability in current environment (if torch is available)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            print("⚠️  Warning: CUDA not available in current environment")
    except ImportError:
        # torch not available in current environment, but that's OK
        # The CLD conda environment should have torch installed
        pass
    
    # Find conda initialization script (common locations)
    conda_init_paths = [
        Path("/opt/anaconda3/etc/profile.d/conda.sh"),
        Path("/opt/conda/etc/profile.d/conda.sh"),
        Path("/usr/local/anaconda3/etc/profile.d/conda.sh"),
        Path("/usr/local/conda/etc/profile.d/conda.sh"),
        Path.home() / "anaconda3/etc/profile.d/conda.sh",
        Path.home() / "miniconda3/etc/profile.d/conda.sh",
        Path("/workspace/miniconda3/etc/profile.d/conda.sh"),
    ]
    
    # Try to find conda.sh
    conda_sh = None
    for path in conda_init_paths:
        if path.exists():
            conda_sh = path
            break
    
    # Build shell command: source conda.sh && conda activate <env> && python <script> && conda deactivate
    start_from_arg = ""
    if start_from > 0:
        start_from_arg = f' --start_from {start_from}'
    
    if conda_sh:
        shell_cmd = f'source "{conda_sh}" && conda activate {conda_env} && python "{cld_script}" --config_path "{cld_infer_config_path}"{start_from_arg} && conda deactivate'
    else:
        # Fallback: try to use conda shell hook (works if conda is in PATH)
        shell_cmd = f'eval "$(conda shell.bash hook)" && conda activate {conda_env} && python "{cld_script}" --config_path "{cld_infer_config_path}"{start_from_arg} && conda deactivate'
    
    print("=" * 60)
    print("STEP 4: CLD Inference")
    print("=" * 60)
    print(f"🔧 Running: bash -c \"conda activate {conda_env} && python ...\"")
    print(f"   Environment: {conda_env}")
    print(f"   Pipeline config: {pipeline_config_path}")
    print(f"   CLD inference config: {cld_infer_config_path}")
    if start_from > 0:
        print(f"   Starting from file index: {start_from}")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"   CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print()
    
    # Run command using bash -c
    try:
        result = subprocess.run(
            ["bash", "-c", shell_cmd],
            check=True,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
        )
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
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start processing from the Nth JSON file (0-indexed). Useful for resuming interrupted runs. (default: 0)"
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
        conda_env=args.conda_env,
        start_from=args.start_from
    )
    sys.exit(exit_code)

