#!/usr/bin/env python
"""
Step 3.5: Generate VLM captions for CLD JSON files using LLaVA codebase.

This script runs in a separate environment (with LLaVA dependencies) to generate
captions for images and update the CLD JSON files with whole_caption field.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml
import torch
from PIL import Image

# Add LLaVA directory to path
LLAVA_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "llava"
if str(LLAVA_DIR) not in sys.path:
    sys.path.insert(0, str(LLAVA_DIR))

try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
    )
except ImportError as e:
    print(f"❌ Failed to import LLaVA modules: {e}")
    print(f"   Make sure you're in the correct conda environment (e.g., llava15)")
    print(f"   and that LLaVA codebase is available at: {LLAVA_DIR}")
    sys.exit(1)


def resolve_path(path: str, config_path: Path) -> Path:
    """Resolve relative paths relative to config file location."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


class LLaVACaptioner:
    """LLaVA captioner using the official LLaVA codebase."""
    
    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        device: str = "cuda",
        load_4bit: bool = False,
        load_8bit: bool = False,
        prompt: str = "Describe style, main subject, and especially the background of the whole image in one short sentence.",
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        num_beams: int = 1,
    ):
        self.device = device
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        
        # Disable torch init for faster loading
        disable_torch_init()
        
        # Get model name from path
        model_name = get_model_name_from_path(model_path)
        
        # Load model
        print(f"📥 Loading LLaVA model from {model_path}...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device_map="auto" if device == "cuda" else {"": device},
            device=device,
        )
        
        # Determine conversation mode
        if "llama-2" in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"
        
        print(f"✅ LLaVA model loaded successfully (conv_mode: {self.conv_mode})")
    
    @torch.inference_mode()
    def generate(self, image: Image.Image) -> str:
        """Generate caption for an image."""
        import re
        
        # Prepare query with image token
        qs = self.prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        # Get conversation template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Process image
        image_sizes = [image.size]
        images_tensor = process_images(
            [image],
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        
        # Tokenize
        input_ids = (
            tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        # ---- Robust decode/parsing (LLaVA checkpoints can differ in whether generate() returns
        # the full sequence [prompt + answer] or only the newly generated tokens) ----
        input_token_len = int(input_ids.shape[1])
        out_token_len = int(output_ids.shape[1])

        # If output is longer than input, we can safely slice out only the new tokens.
        # Otherwise, fall back to decoding the whole output (some wrappers return only new tokens).
        if out_token_len > input_token_len:
            decoded = self.tokenizer.decode(
                output_ids[0, input_token_len:],
                skip_special_tokens=True,
            )
        else:
            decoded = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            )

        text = decoded.strip()

        # If still empty, try decoding without skipping special tokens (helps diagnose EOS-only outputs),
        # then strip common special tokens manually.
        if not text:
            decoded_raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=False).strip()
            text = decoded_raw.replace("</s>", "").replace("<s>", "").strip()

        # Parse out the assistant answer if the decoded text includes roles.
        # For llava_v1 roles are ("USER", "ASSISTANT")
        # The prompt often ends with "ASSISTANT:" and some generations include it again.
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[1].strip()
        elif "Assistant:" in text:
            text = text.split("Assistant:", 1)[1].strip()

        # Remove any trailing separators if present as literal strings.
        if conv.sep_style == SeparatorStyle.TWO and conv.sep2:
            if text.endswith(conv.sep2):
                text = text[: -len(conv.sep2)].strip()
        if conv.sep and text.endswith(conv.sep):
            text = text[: -len(conv.sep)].strip()

        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--force", "-f", action="store_true", help="Force regenerate captions even if they already exist")
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Step 3.5 config
    cld_output_dir = resolve_path(config["step3_5"]["cld_output_dir"], config_path)
    vlm_config = config["step3_5"]["vlm"]
    # Command line --force overrides config
    force_regenerate = args.force or config["step3_5"].get("force_regenerate", False)
    
    if force_regenerate:
        print("🔄 Force regenerate mode: will overwrite existing captions")
    
    if not vlm_config.get("use_vlm_caption", False):
        print("ℹ️  VLM caption generation disabled (use_vlm_caption=false)")
        return
    
    # Initialize LLaVA captioner
    print("🚀 Initializing LLaVA captioner...")
    try:
        vlm = LLaVACaptioner(
            model_path=vlm_config.get("vlm_model_id", "liuhaotian/llava-v1.5-7b"),
            model_base=vlm_config.get("vlm_model_base", None),
            device=vlm_config.get("vlm_device", "cuda"),
            load_4bit=vlm_config.get("vlm_load_in_4bit", True),
            load_8bit=vlm_config.get("vlm_load_in_8bit", False),
            prompt=vlm_config.get("vlm_prompt", "Describe style, main subject, and especially the background of the whole image in one short sentence."),
            max_new_tokens=vlm_config.get("vlm_max_new_tokens", 96),
            temperature=vlm_config.get("vlm_temperature", 0.2),
            top_p=vlm_config.get("vlm_top_p", None),
            num_beams=vlm_config.get("vlm_num_beams", 1),
        )
        print("✅ LLaVA captioner initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLaVA captioner: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Find all JSON files
    json_files = sorted(list(Path(cld_output_dir).glob("*.json")))
    if not json_files:
        print(f"❌ No JSON files found in {cld_output_dir}")
        sys.exit(1)
    
    print(f"📂 Found {len(json_files)} JSON files to process")
    
    processed = 0
    failed = 0
    
    for json_file in json_files:
        try:
            # Load JSON
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check if caption already exists (unless force_regenerate is True)
            if not force_regenerate and data.get("whole_caption") and data["whole_caption"].strip():
                print(f"  ⏭️  Skipping {json_file.name}: caption already exists")
                continue
            
            # If force regenerate, clear existing caption
            if force_regenerate and data.get("whole_caption"):
                print(f"  🔄 Regenerating caption for {json_file.name} (clearing existing caption)")
                data["whole_caption"] = ""
            
            # Load image
            image_path = Path(data["image_path"])
            if not image_path.exists():
                print(f"  ⚠️  Image not found: {image_path}")
                failed += 1
                continue
            
            img = Image.open(image_path).convert("RGB")
            
            # Generate caption
            caption = vlm.generate(img)
            
            # Debug: print caption for first few files or if empty/short
            if processed < 5 or not caption or len(caption.strip()) < 10:
                print(f"  📝 Generated caption for {json_file.name}: '{caption[:200]}'")
                print(f"     Length: {len(caption)} chars")
                if not caption or len(caption.strip()) < 10:
                    print(f"     ⚠️  Warning: Caption is empty or very short!")
            
            # Ensure caption is not None
            if caption is None:
                caption = ""
            
            # Update JSON
            data["whole_caption"] = caption
            if not data.get("caption") or not data.get("caption").strip():
                data["caption"] = caption
            
            # Save updated JSON (use absolute path to ensure we save to the right location)
            json_file_abs = Path(json_file).resolve()
            with open(json_file_abs, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Verify the save worked
            if processed < 3:
                with open(json_file_abs, "r", encoding="utf-8") as f:
                    verify_data = json.load(f)
                    if verify_data.get("whole_caption") != caption:
                        print(f"     ⚠️  Warning: Caption mismatch after save!")
                    else:
                        print(f"     ✅ Caption saved successfully")
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{len(json_files)} files...")
            elif processed <= 3:  # Print first few captions for debugging
                print(f"  ✅ Generated caption for {json_file.name}: '{caption[:100]}...'")
        
        except Exception as e:
            print(f"  ❌ Failed to process {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"\n✅ Completed: {processed} processed, {failed} failed")
    print(f"📂 Updated JSON files in {cld_output_dir}")


if __name__ == "__main__":
    main()


