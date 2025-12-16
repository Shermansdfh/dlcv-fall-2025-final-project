from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def _snapshot_download(
    repo_id: str,
    local_dir: Path,
    *,
    token: str | None,
    allow_patterns: list[str] | None = None,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download CLD/FLUX related assets to checkpoints/ and generate configs/cld/default.yaml（based on repo root）."
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token（can also use environment variables HF_TOKEN or HUGGINGFACE_HUB_TOKEN）.",
    )
    parser.add_argument("--skip-flux", action="store_true", help="跳過下載 black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--skip-adapter",
        action="store_true",
        help="Skip downloading alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    )
    parser.add_argument("--skip-cld", action="store_true", help="Skip downloading thuteam/CLD（LoRA/trans_vae）")
    args = parser.parse_args()

    # scripts/download_cld_assets.py
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # scripts/ -> repo root（关键）

    # checkpoints/
    checkpoints_dir = repo_root / "checkpoints"
    flux_root = checkpoints_dir / "flux"
    cld_ckpt_dir = checkpoints_dir / "cld"

    # checkpoints/flux/*
    flux_path = flux_root / "FLUX.1-dev"
    adapter_path = flux_root / "FLUX.1-dev-Controlnet-Inpainting-Alpha"

    # configs/cld/default.yaml
    config_dir = repo_root / "configs" / "cld"
    config_path = config_dir / "default.yaml"

    print(f"REPO_ROOT: {repo_root}")
    print(f"CHECKPOINT_DIR: {checkpoints_dir}")
    print(f"CONFIG: {config_path}")

    # ==========================================
    # 1) Download weights to checkpoints/
    # ==========================================
    hf_token = args.hf_token
    if not hf_token:
        print("\n⚠️  NOTE: No HF token provided（HF_TOKEN / HUGGINGFACE_HUB_TOKEN / --hf-token）.")
        print("   FLUX.1-dev may require token and authorization; if download fails, please set the token first.")

    if not args.skip_flux:
        print("\n--- [a] Downloading FLUX.1-dev ---")
        print(f"   Target: {flux_path}")
        _snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir=flux_path,
            token=hf_token,
        )
        print(f"✅ FLUX saved to: {flux_path}")

    if not args.skip_adapter:
        print("\n--- [b] Downloading Adapter weights ---")
        print(f"   Target: {adapter_path}")
        _snapshot_download(
            repo_id="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
            local_dir=adapter_path,
            token=hf_token,
        )
        print(f"✅ Adapter saved to: {adapter_path}")

    if not args.skip_cld:
        print("\n--- [c] Downloading CLD LoRA + transp_vae ---")
        print(f"   Target: {cld_ckpt_dir}")
        cld_patterns = [
            "decouple_LoRA/*",
            "pre_trained_LoRA/*",
            "prism_ft_LoRA/*",
            "trans_vae/*",
        ]
        _snapshot_download(
            repo_id="thuteam/CLD",
            local_dir=cld_ckpt_dir,
            token=hf_token,
            allow_patterns=cld_patterns,
        )
        print(f"✅ CLD weights saved to: {cld_ckpt_dir}")

    # ==========================================
    # 3) Generate YAML configuration (pipeline 層)
    # ==========================================
    print("\n--- [d] Generating configs/cld/default.yaml ---")
    config_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "pretrained_model_name_or_path": str(flux_path),
        "pretrained_adapter_path": str(adapter_path),
        "transp_vae_path": str(cld_ckpt_dir / "trans_vae" / "0008000.pt"),
        "pretrained_lora_dir": str(cld_ckpt_dir / "pre_trained_LoRA"),
        "artplus_lora_dir": str(cld_ckpt_dir / "prism_ft_LoRA"),
        "lora_ckpt": str(cld_ckpt_dir / "decouple_LoRA" / "transformer"),
        "layer_ckpt": str(cld_ckpt_dir / "decouple_LoRA"),
        "adapter_lora_dir": str(cld_ckpt_dir / "decouple_LoRA" / "adapter"),
    }

    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

    print(f"✅ Config created: {config_path}")
    print("Content preview:")
    print("-" * 20)
    print(yaml.dump(config_data, sort_keys=False))
    print("-" * 20)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


