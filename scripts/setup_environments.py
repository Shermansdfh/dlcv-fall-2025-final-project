#!/usr/bin/env python
"""
Setup script for all pipeline environments.

This script sets up:
1. CLD conda environment (from environment.yml)
2. LayerD uv environment
3. Ultralytics conda environment
4. LLaVA conda environment
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"🔧 Running: {' '.join(cmd)}")
    if cwd:
        print(f"   Working directory: {cwd}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=check,
            capture_output=False,  # Show output in real-time
            text=True,
        )
        return result
    except FileNotFoundError as e:
        print(f"❌ Command not found: {cmd[0]}")
        print(f"   Please ensure {cmd[0]} is installed and in PATH")
        raise
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        raise


def check_conda() -> bool:
    """Check if conda is available."""
    try:
        subprocess.run(["conda", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def check_uv() -> bool:
    """Check if uv is available."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def setup_cld(force: bool = False) -> int:
    """Setup CLD conda environment."""
    print("\n" + "=" * 60)
    print("Setting up CLD conda environment")
    print("=" * 60)
    
    if not check_conda():
        print("❌ Conda not found. Please install conda first.")
        return 1
    
    env_yml = REPO_ROOT / "third_party" / "cld" / "environment.yml"
    if not env_yml.exists():
        print(f"❌ CLD environment.yml not found: {env_yml}")
        return 1
    
    # Check if environment already exists
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        env_exists = "CLD" in result.stdout
    except subprocess.CalledProcessError:
        env_exists = False
    
    if env_exists and not force:
        print("ℹ️  CLD conda environment already exists. Use --force to recreate.")
        response = input("   Do you want to recreate it? (y/N): ")
        if response.lower() != "y":
            print("   Skipping CLD environment setup.")
            return 0
    
    if env_exists and force:
        print("🗑️  Removing existing CLD environment...")
        try:
            run_command(["conda", "env", "remove", "-n", "CLD", "-y"], check=False)
        except subprocess.CalledProcessError:
            pass  # Ignore if removal fails
    
    print(f"📦 Creating CLD conda environment from {env_yml}...")
    try:
        run_command(["conda", "env", "create", "-f", str(env_yml)])
        print("✅ CLD environment created successfully")
        return 0
    except subprocess.CalledProcessError:
        print("❌ Failed to create CLD environment")
        return 1


def setup_layerd(force: bool = False) -> int:
    """Setup LayerD uv environment."""
    print("\n" + "=" * 60)
    print("Setting up LayerD uv environment")
    print("=" * 60)
    
    if not check_uv():
        print("❌ uv not found. Please install uv first.")
        print("   Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1
    
    layerd_dir = REPO_ROOT / "third_party" / "layerd"
    if not layerd_dir.exists():
        print(f"❌ LayerD directory not found: {layerd_dir}")
        print("   Please ensure third_party/layerd submodule is initialized")
        return 1
    
    pyproject_toml = layerd_dir / "pyproject.toml"
    if not pyproject_toml.exists():
        print(f"❌ LayerD pyproject.toml not found: {pyproject_toml}")
        return 1
    
    print(f"📦 Syncing LayerD uv environment in {layerd_dir}...")
    try:
        run_command(["uv", "sync"], cwd=layerd_dir)
        print("✅ LayerD uv environment synced successfully")
        return 0
    except subprocess.CalledProcessError:
        print("❌ Failed to sync LayerD environment")
        return 1


def setup_ultralytics(force: bool = False) -> int:
    """Setup Ultralytics conda environment."""
    print("\n" + "=" * 60)
    print("Setting up Ultralytics conda environment")
    print("=" * 60)
    
    if not check_conda():
        print("❌ Conda not found. Please install conda first.")
        return 1
    
    # Check if environment already exists
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        env_exists = "ultralytics" in result.stdout
    except subprocess.CalledProcessError:
        env_exists = False
    
    if env_exists and not force:
        print("ℹ️  Ultralytics conda environment already exists. Use --force to recreate.")
        response = input("   Do you want to recreate it? (y/N): ")
        if response.lower() != "y":
            print("   Skipping Ultralytics environment setup.")
            return 0
    
    if env_exists and force:
        print("🗑️  Removing existing ultralytics environment...")
        try:
            run_command(["conda", "env", "remove", "-n", "ultralytics", "-y"], check=False)
        except subprocess.CalledProcessError:
            pass  # Ignore if removal fails
    
    print("📦 Creating ultralytics conda environment...")
    try:
        # Create environment with Python (default version)
        run_command(["conda", "create", "-n", "ultralytics", "python", "-y"])
        
        # Install ultralytics
        print("📦 Installing ultralytics package...")
        run_command([
            "conda", "run",
            "-n", "ultralytics",
            "pip", "install", "ultralytics"
        ])
        
        print("✅ Ultralytics environment created successfully")
        return 0
    except subprocess.CalledProcessError:
        print("❌ Failed to create Ultralytics environment")
        return 1


def setup_llava(force: bool = False) -> int:
    """Setup LLaVA conda environment."""
    print("\n" + "=" * 60)
    print("Setting up LLaVA conda environment")
    print("=" * 60)
    
    if not check_conda():
        print("❌ Conda not found. Please install conda first.")
        return 1
    
    llava_dir = REPO_ROOT / "third_party" / "llava"
    if not llava_dir.exists():
        print(f"❌ LLaVA directory not found: {llava_dir}")
        print("   Please ensure third_party/llava submodule is initialized")
        return 1
    
    # Check for either pyproject.toml or setup.py (both work with pip install -e .)
    pyproject_toml = llava_dir / "pyproject.toml"
    setup_py = llava_dir / "setup.py"
    if not pyproject_toml.exists() and not setup_py.exists():
        print(f"❌ LLaVA package file not found: neither pyproject.toml nor setup.py exists")
        print(f"   Expected one of: {pyproject_toml} or {setup_py}")
        return 1
    
    # Check if environment already exists
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        env_exists = "llava" in result.stdout
    except subprocess.CalledProcessError:
        env_exists = False
    
    if env_exists and not force:
        print("ℹ️  LLaVA conda environment already exists. Use --force to recreate.")
        response = input("   Do you want to recreate it? (y/N): ")
        if response.lower() != "y":
            print("   Skipping LLaVA environment setup.")
            return 0
    
    if env_exists and force:
        print("🗑️  Removing existing llava environment...")
        try:
            run_command(["conda", "env", "remove", "-n", "llava", "-y"], check=False)
        except subprocess.CalledProcessError:
            pass  # Ignore if removal fails
    
    print("📦 Creating llava conda environment with Python 3.10...")
    try:
        # Create environment with Python 3.10
        run_command(["conda", "create", "-n", "llava", "python=3.10", "-y"])
        
        # Upgrade pip
        print("📦 Upgrading pip...")
        run_command([
            "conda", "run",
            "-n", "llava",
            "pip", "install", "--upgrade", "pip"
        ])
        
        # Install LLaVA in editable mode
        print(f"📦 Installing LLaVA from {llava_dir}...")
        run_command([
            "conda", "run",
            "-n", "llava",
            "pip", "install", "-e", "."
        ], cwd=llava_dir)
        
        print("✅ LLaVA environment created successfully")
        return 0
    except subprocess.CalledProcessError:
        print("❌ Failed to create LLaVA environment")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Setup all pipeline environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup all environments
  python scripts/setup_environments.py --all

  # Setup specific environments
  python scripts/setup_environments.py --cld --ultralytics

  # Force recreate existing environments
  python scripts/setup_environments.py --all --force
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Setup all environments"
    )
    parser.add_argument(
        "--cld",
        action="store_true",
        help="Setup CLD conda environment"
    )
    parser.add_argument(
        "--layerd",
        action="store_true",
        help="Setup LayerD uv environment"
    )
    parser.add_argument(
        "--ultralytics",
        action="store_true",
        help="Setup Ultralytics conda environment"
    )
    parser.add_argument(
        "--llava",
        action="store_true",
        help="Setup LLaVA conda environment"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate existing environments"
    )
    
    args = parser.parse_args()
    
    # If --all is specified, set all flags
    if args.all:
        args.cld = True
        args.layerd = True
        args.ultralytics = True
        args.llava = True
    
    # If no specific environment is selected, show help
    if not any([args.cld, args.layerd, args.ultralytics, args.llava]):
        parser.print_help()
        return 1
    
    print("🚀 Starting environment setup...")
    print(f"   Repository root: {REPO_ROOT}")
    print()
    
    results = {}
    
    if args.cld:
        results["CLD"] = setup_cld(force=args.force)
    
    if args.layerd:
        results["LayerD"] = setup_layerd(force=args.force)
    
    if args.ultralytics:
        results["Ultralytics"] = setup_ultralytics(force=args.force)
    
    if args.llava:
        results["LLaVA"] = setup_llava(force=args.force)
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    all_success = True
    for env_name, exit_code in results.items():
        status = "✅ Success" if exit_code == 0 else "❌ Failed"
        print(f"  {env_name}: {status}")
        if exit_code != 0:
            all_success = False
    
    if all_success:
        print("\n🎉 All environments set up successfully!")
        return 0
    else:
        print("\n⚠️  Some environments failed to set up. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

