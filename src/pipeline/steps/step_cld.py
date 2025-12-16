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
    
    # Check CUDA availability in current environment (if torch is available)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            print("⚠️  Warning: CUDA not available in current environment")
            print("   CLD inference requires GPU. Please ensure:")
            print("   1. GPU is available and drivers are installed")
            print("   2. PyTorch with CUDA support is installed in the CLD conda environment")
            print("   3. CUDA_VISIBLE_DEVICES is set correctly (if using specific GPU)")
    except ImportError:
        # torch not available in current environment, but that's OK
        # The CLD conda environment should have torch installed
        pass
    
    # Prepare environment variables to pass to subprocess
    # conda run may reset some environment variables, so we need to ensure CUDA-related ones are preserved
    env = os.environ.copy()
    
    # Collect CUDA-related environment variables that need to be preserved
    cuda_env_vars = {}
    cuda_env_var_names = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "LD_LIBRARY_PATH",  # May contain CUDA library paths
    ]
    for var in cuda_env_var_names:
        if var in os.environ:
            cuda_env_vars[var] = os.environ[var]
            env[var] = os.environ[var]
    
    # Build command: conda run -n <env> python <script> --config_path <cld_infer_config>
    # Note: conda run may not properly expose CUDA devices, so we pass env vars explicitly
    cmd = [
        "conda", "run",
        "-n", conda_env,
        "--no-capture-output",
        "python", str(cld_script),
        "--config_path", str(cld_infer_config_path)
    ]
    
    # Pass CUDA-related environment variables to subprocess
    # This ensures conda run can access CUDA devices
    if cuda_env_vars:
        print(f"   Passing CUDA environment variables: {', '.join(cuda_env_vars.keys())}")
    
    print("=" * 60)
    print("STEP 4: CLD Inference")
    print("=" * 60)
    if cuda_env_vars:
        print(f"🔧 Running: bash -c \"... conda run ...\"")
        print(f"   (with CUDA environment variables: {', '.join(cuda_env_vars.keys())})")
    else:
        print(f"🔧 Running: {' '.join(cmd)}")
    print(f"   Environment: {conda_env}")
    print(f"   Pipeline config: {pipeline_config_path}")
    print(f"   CLD inference config: {cld_infer_config_path}")
    if "CUDA_VISIBLE_DEVICES" in cuda_env_vars:
        print(f"   CUDA_VISIBLE_DEVICES: {cuda_env_vars['CUDA_VISIBLE_DEVICES']}")
    print()
    
    # Run command with environment variables
    try:
        result = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env, shell=bool(cuda_env_vars))
        print("\n✅ Step 4 (CLD Inference) completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step 4 (CLD Inference) failed with exit code {e.returncode}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Verify GPU is available: nvidia-smi")
        print("   2. Check CUDA in CLD environment:")
        print(f"      conda run -n {conda_env} python -c \"import torch; print(f'CUDA available: {{torch.cuda.is_available()}}, Devices: {{torch.cuda.device_count()}}')\"")
        print("   3. If CUDA is False or device_count is 0 in conda run but True in conda activate:")
        print("      This is a known issue with 'conda run' - CUDA devices may not be accessible.")
        print("      Solution: Use 'conda activate' instead:")
        print(f"      conda activate {conda_env}")
        print(f"      python {cld_script} --config_path {cld_infer_config_path}")
        print("   4. Ensure CUDA_VISIBLE_DEVICES is set correctly (if using specific GPU)")
        print("   5. Check that PyTorch with CUDA support is installed in CLD environment")
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

