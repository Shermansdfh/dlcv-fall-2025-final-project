from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _find_repo_root(start: Path) -> Path:
    """
    Find repo root from the current file location.
    Condition: contains third_party/cld/infer/infer.py.
    """
    for p in [start, *start.parents]:
        if (p / "third_party" / "cld" / "infer" / "infer.py").exists():
            return p
    # fallback: assume src/cld_generation/infer_dlcv.py -> repo_root is parents[2]
    return start.parents[2]


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    """
    Load Python module from file path, avoid the problem that third_party/cld doesn't have package __init__.py.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project-level CLD inference runner.\n"
            "This script is placed in src/cld_generation, is a wrapper: reuse third_party/cld/infer/infer.py as much as possible, "
            "avoid copying large model initialization and inference logic."
        )
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default=None,
        help="YAML config path（e.g. configs/exp001/cld/infer.yaml）.",
    )
    parser.add_argument(
        "--cld_infer_py",
        type=str,
        default=None,
        help="third_party CLD's infer.py path（default: <repo_root>/third_party/cld/infer/infer.py）.",
    )
    args = parser.parse_args()

    # Don't hardcode CUDA_VISIBLE_DEVICES here, leave it to scheduler/launcher
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Check CUDA availability before proceeding
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        
        if not cuda_available:
            print("❌ CUDA is not available in this environment")
            print("   CLD inference requires GPU. Please ensure:")
            print("   1. GPU is available: nvidia-smi")
            print("   2. PyTorch with CUDA support is installed")
            print("   3. CUDA_VISIBLE_DEVICES is set correctly (if using specific GPU)")
            print("\n   If you're using conda run, try activating the environment directly:")
            print(f"   conda activate CLD")
            print(f"   python {Path(__file__).resolve()} --config_path <config>")
            return 1
        
        if cuda_device_count == 0:
            print("❌ CUDA is available but no devices are visible")
            print("   This can happen when using 'conda run' - CUDA devices may not be accessible.")
            print("\n   Solutions:")
            print("   1. Use 'conda activate' instead of 'conda run':")
            print(f"      conda activate CLD")
            print(f"      python {Path(__file__).resolve()} --config_path <config>")
            print("   2. Or ensure CUDA_VISIBLE_DEVICES is set before conda run:")
            print("      export CUDA_VISIBLE_DEVICES=0")
            print("      conda run -n CLD python ...")
            print("   3. Check GPU availability:")
            print("      nvidia-smi")
            return 1
        
        print(f"✅ CUDA available: {cuda_device_count} device(s)")
    except ImportError:
        print("⚠️  Warning: torch not found. CUDA availability cannot be checked.")
        print("   Proceeding anyway, but CLD inference will likely fail without GPU.")

    this_file = Path(__file__).resolve()
    repo_root = _find_repo_root(this_file.parent)

    # Default config
    default_cfg = repo_root / "configs" / "exp001" / "cld" / "infer.yaml"
    config_path = Path(args.config_path).expanduser() if args.config_path else default_cfg
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    # Load third_party cld infer.py
    cld_infer_py = Path(args.cld_infer_py).expanduser() if args.cld_infer_py else (repo_root / "third_party" / "cld" / "infer" / "infer.py")
    if not cld_infer_py.is_absolute():
        cld_infer_py = (repo_root / cld_infer_py).resolve()
    if not cld_infer_py.exists():
        raise FileNotFoundError(f"CLD infer.py not found: {cld_infer_py}")

    cld_root = cld_infer_py.parent.parent  # .../third_party/cld
    if str(cld_root) not in sys.path:
        sys.path.insert(0, str(cld_root))

    # Important: CLD infer.py uses `from models...` / `from tools...`, need to set cwd / sys.path to cld_root
    # Here we only change cwd to cld_root（same as finals/CLD/infer/infer_dlcv.py）, ensure consistent relative path/resource reading.
    os.chdir(str(cld_root))

    cld_infer = _load_module_from_path("cld_infer", cld_infer_py)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Final CUDA check and fix before calling inference_layout
    # This ensures CUDA is properly initialized in the current process
    try:
        import torch
        
        if torch.cuda.is_available() and torch.cuda.device_count() == 0:
            print("\n⚠️  Warning: CUDA is available but no devices are visible.")
            print("   This may happen with 'conda run' or in Docker containers without GPU access.")
            
            # Try to initialize CUDA by creating a tensor
            try:
                _ = torch.zeros(1).cuda()
                print("   ✅ CUDA context initialized successfully")
            except RuntimeError as e:
                print(f"   ❌ Failed to initialize CUDA context: {e}")
                print("\n   Solutions:")
                print("   1. In Docker: Ensure GPU is properly mounted:")
                print("      docker run --gpus all ...")
                print("      # or with nvidia-docker:")
                print("      docker run --runtime=nvidia ...")
                print("   2. Check GPU availability:")
                print("      nvidia-smi")
                print("   3. Set CUDA_VISIBLE_DEVICES before running:")
                print("      export CUDA_VISIBLE_DEVICES=0")
                print("   4. Use 'conda activate' instead of 'conda run':")
                print("      conda activate CLD")
                print(f"      python {Path(__file__).resolve()} --config_path <config>")
                return 1
            
            # Monkey patch torch.load to handle device_count() == 0 case
            # This fixes the issue where third_party/cld/infer/infer.py uses
            # torch.load(..., map_location=torch.device("cuda")) when device_count() == 0
            # We'll load on CPU first, then move to CUDA if device becomes available
            original_torch_load = torch.load
            
            def patched_torch_load(*args, **kwargs):
                """Patched torch.load that handles CUDA device issues."""
                # If map_location is torch.device("cuda") and device_count() == 0,
                # change it to load on CPU first
                if "map_location" in kwargs:
                    map_loc = kwargs["map_location"]
                    if isinstance(map_loc, torch.device) and map_loc.type == "cuda":
                        if torch.cuda.device_count() == 0:
                            # Load on CPU first
                            kwargs["map_location"] = "cpu"
                            result = original_torch_load(*args, **kwargs)
                            # Try to move to CUDA if device becomes available
                            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                                try:
                                    if isinstance(result, dict):
                                        return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
                                    elif isinstance(result, torch.Tensor):
                                        return result.cuda()
                                except Exception:
                                    # If CUDA still not available, return CPU version
                                    pass
                            return result
                return original_torch_load(*args, **kwargs)
            
            # Apply monkey patch
            torch.load = patched_torch_load
            print("   🔧 Applied monkey patch for torch.load to handle CUDA device issues")
            
    except ImportError:
        pass  # torch check already done above

    # Load config first to check if we should use pipeline dataset
    config = cld_infer.load_config(str(config_path))
    
    # Check if we should use pipeline dataset instead of LayoutTrainDataset
    use_pipeline_dataset = config.get("use_pipeline_dataset", False)
    
    if use_pipeline_dataset:
        # Import PipelineDataset from our custom dataset module
        # Need to add repo_root/src to path to import our custom dataset
        repo_root_src = repo_root / "src"
        if str(repo_root_src) not in sys.path:
            sys.path.insert(0, str(repo_root_src))
        
        from data.custom_cld_dataset import PipelineDataset, collate_fn_pipeline
        
        # Monkey patch tools.dataset module to replace LayoutTrainDataset with PipelineDataset
        # This way, when infer.py does `from tools.dataset import LayoutTrainDataset`,
        # it will actually get our PipelineDataset
        import tools.dataset as dataset_module
        
        # Store original LayoutTrainDataset for fallback
        original_LayoutTrainDataset = dataset_module.LayoutTrainDataset
        original_collate_fn = dataset_module.collate_fn
        
        # Create a closure to capture config for max_image_side and max_image_size
        def create_patched_LayoutTrainDataset(config_ref):
            """Factory function to create a patched LayoutTrainDataset that uses PipelineDataset."""
            class LayoutTrainDatasetWrapper:
                """Wrapper that makes PipelineDataset compatible with LayoutTrainDataset interface."""
                def __init__(self, data_dir, split="test"):
                    # Ignore split parameter (PipelineDataset doesn't use it)
                    # Get max_image_side and max_image_size from config
                    self.dataset = PipelineDataset(
                        data_dir=data_dir,
                        max_image_side=config_ref.get('max_image_side'),
                        max_image_size=config_ref.get('max_image_size'),
                    )
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    return self.dataset[idx]
            
            return LayoutTrainDatasetWrapper
        
        # Replace LayoutTrainDataset with patched version that uses config
        dataset_module.LayoutTrainDataset = create_patched_LayoutTrainDataset(config)
        dataset_module.collate_fn = collate_fn_pipeline
        
        print("✅ Patched tools.dataset to use PipelineDataset instead of LayoutTrainDataset")
    
    # Call inference_layout (will use patched dataset if use_pipeline_dataset=True)
    cld_infer.inference_layout(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


