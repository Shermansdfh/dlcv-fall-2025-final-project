# DLCV Final Project - Layout Decomposition Pipeline

A complete end-to-end image layout decomposition pipeline from object detection to layered composition.

## Important Links

- [CLD Repo](https://github.com/monkek123King/CLD/?tab=readme-ov-file)
- [CLD HuggingFace](https://huggingface.co/thuteam/CLD)
- [LayerD Repo](https://github.com/CyberAgentAILab/LayerD)
- [RTDETR](https://github.com/ultralytics/ultralytics)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Crello Dataset](https://huggingface.co/datasets/cyberagent/crello)
- [TA Dataset](https://huggingface.co/datasets/WalkerHsu/DLCV2025_final_project_piccollage)

## System Used

- **GPU**: NVIDIA H100 80GB
- **Storage**: 100GB+ recommended
- **CUDA**: Compatible CUDA version for PyTorch

---

## Environment Setup

### Quick Setup

```bash
# Clone repository and initialize submodules
# git clone <repository-url>
# cd dlcv-fall-2025-final-project
# git submodule update --init --recursive  

# Setup all environments
python scripts/setup_environments.py --all
```

### Environment Overview

| Environment | Type | Name | Purpose |
|------------|------|------|---------|
| CLD | Conda | `CLD` | CLD format conversion and inference |
| LayerD | uv | - | LayerD layer decomposition |
| Ultralytics | Conda | `ultralytics` | RTDETR object detection |
| LLaVA | Conda | `llava` | VLM caption generation |

### Setup Specific Environments

```bash
# Setup individual environments
python scripts/setup_environments.py --cld --ultralytics --llava --layerd

# Force recreate environments
python scripts/setup_environments.py --all --force
```

### Verify Environments

```bash
# Check conda environments
conda env list

# Test environments
conda run -n CLD python --version
conda run -n ultralytics python -c "import ultralytics; print(ultralytics.__version__)"
conda run -n llava python --version

# Test LayerD (requires cd to directory)
cd third_party/layerd
uv run python --version
```

**Detailed setup guide**: [scripts/README_SETUP.md](scripts/README_SETUP.md)

---

## Checkpoints and Model Downloads

### RTDETR Checkpoint

Download pre-trained RTDETR checkpoint:

```bash
# Install gdown if needed
pip install gdown

# Create directory
mkdir -p checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights

# Download checkpoint
gdown --id 1TT5iBr1ber8pT0E7tcfUE-FV1ssn4dcQ -O checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt
```

### CLD Assets

Download CLD models and weights:

```bash
# Set HuggingFace token (required for some models)
export HF_TOKEN="your_huggingface_token_here"

# Download CLD assets
python scripts/download_cld_assets.py
```

**Get HuggingFace Token**: [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)

Downloaded assets include:
- FLUX.1-dev model
- ControlNet Inpainting Alpha adapter
- CLD LoRA weights
- Transparent VAE weights

### Testing Data

```bash
# Download test data from Google Drive
python scripts/download_testing_data.py --file-id "YOUR_FILE_ID"
```

---

## Pipeline Usage

### Overview

The pipeline consists of 5 main steps:

1. **RTDETR Detection** - Detect objects in images
2. **LayerD Decomposition** - Decompose images into layers
3. **CLD Format Conversion** - Convert results to CLD format
4. **VLM Caption Generation** (Optional) - Generate image captions
5. **CLD Inference** - Final layered composition

### Configuration

Create or copy a configuration file:

```bash
cp configs/exp001/pipeline.yaml configs/my_experiment/pipeline.yaml
# Edit the configuration file to set input/output paths
```

### Execute Pipeline

```bash
# Step 1: RTDETR Detection
python -m src.pipeline.steps.step_rtdetr --config configs/my_experiment/pipeline.yaml

# Step 2: LayerD Decomposition
python -m src.pipeline.steps.step_layerd --config configs/my_experiment/pipeline.yaml

# Step 3: CLD Format Conversion
python -m src.pipeline.steps.step_conversion --config configs/my_experiment/pipeline.yaml

# Step 3.5: VLM Caption Generation (Optional)
python -m src.pipeline.steps.step_vlm --config configs/my_experiment/pipeline.yaml

# Step 4: CLD Inference
python -m src.pipeline.steps.step_cld --config configs/my_experiment/pipeline.yaml
```

### Pipeline Configuration Example

```yaml
# RTDETR Detection Step
rtdetr:
  input_dir: "/workspace/finals/data/inputs"
  output_dir: "/workspace/finals/outputs/pipeline_outputs/rtdetr"
  model_path: "/workspace/finals/checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt"
  conf: 0.4

# LayerD Decomposition Step
layerd:
  rtdetr_output_dir: "/workspace/finals/outputs/pipeline_outputs/rtdetr"
  output_dir: "/workspace/finals/outputs/pipeline_outputs/layerd"
  max_iterations: 2
  device: "cuda"
  matting_process_size: [512, 512]
  max_image_size: [1536, 1536]

# CLD Format Conversion Step
cld:
  rtdetr_output_dir: "/workspace/finals/outputs/pipeline_outputs/rtdetr"
  layerd_output_dir: "/workspace/finals/outputs/pipeline_outputs/layerd"
  output_dir: "/workspace/finals/outputs/pipeline_outputs/cld"
```

**Detailed pipeline guide**: [PIPELINE_README.md](PIPELINE_README.md)

---

## Model Training

### RTDETR Fine-tuning

#### 1. Prepare Dataset

```bash
# Activate ultralytics environment
conda activate ultralytics

# Prepare DLCV Bounding Box Dataset
python -m src.data.dlcv_bbox_dataset
```

**Dataset Structure**:
```
data/dlcv_bbox_dataset/
├── data.yaml          # YOLO dataset config
├── images/
│   ├── train/         # Training images (90%)
│   └── val/           # Validation images (10%)
└── labels/
    ├── train/         # Training labels
    └── val/           # Validation labels
```

**Dataset Classes**:
- Class 0: `layout_element` - General layout elements
- Class 1: `text` - Text elements

#### 2. Train Model

```bash
# Activate environment
conda activate ultralytics

# Start training
python -m src.bbox.train_rtdetr
```

**Training Configuration**:
- **Model**: RTDETR-L (Large)
- **Epochs**: 100
- **Batch Size**: 16 (reduce to 8 if OOM)
- **Image Size**: 640x640
- **Optimizer**: AdamW
- **Learning Rate**: 0.0001
- **Cache**: 'disk' (prevents OOM)
- **AMP**: True

**Training Time Estimate**:
- 20K images, 100 epochs, batch=16, H100: ~4-6 hours
- 50K images, 100 epochs, batch=16, H100: ~10-15 hours

**Output**: `checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt`

#### 3. Training Strategy

**Disabled Augmentations** (to preserve layout logic):
- ❌ Mosaic (`mosaic=0.0`)
- ❌ Mixup (`mixup=0.0`)
- ❌ Rotation (`degrees=0.0`)

**Enabled Augmentations**:
- ✅ Scaling (`scale=0.5`)
- ✅ Horizontal flip (`fliplr=0.5`)
- ✅ Color variations (`hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`)

#### 4. Customize Training

Edit `src/bbox/train_rtdetr.py` to modify training parameters:

```python
results = model.train(
    data=DATASET_PATH / "data.yaml",
    epochs=150,        # Increase epochs
    batch=8,           # Reduce batch size if OOM
    lr0=0.00005,       # Lower learning rate
    cache='disk',      # Use disk cache (prevents OOM)
    # ... other parameters
)
```

---

## Tools and Scripts

### Visualization

**Bbox Visualization**:
```bash
# Single file
python -m src.bbox.visualize_bbox_gif \
  --input outputs/pipeline_outputs/cld/image1.json \
  --output outputs/pipeline_outputs/cld/image1.gif \
  --use-quantized

# Entire directory
python -m src.bbox.visualize_bbox_gif \
  --input outputs/pipeline_outputs/cld \
  --output-dir outputs/pipeline_outputs/cld_gif \
  --use-quantized
```

### Dataset Preparation

```bash
# Prepare RTDETR training dataset
python -m src.data.dlcv_bbox_dataset
```

---

## Project Structure

```
dlcv-fall-2025-final-project/
├── configs/              # Configuration files
│   └── exp001/          # Experiment configs
│       ├── pipeline.yaml
│       └── cld/
├── src/                  # Source code
│   ├── pipeline/        # Pipeline orchestration
│   ├── bbox/           # RTDETR
│   ├── layerd/         # LayerD
│   ├── adapters/       # Format adapters
│   ├── caption/        # VLM caption
│   ├── cld/            # CLD inference
│   └── data/           # Dataset tools
├── scripts/             # Utility scripts
│   ├── setup_environments.py
│   ├── download_cld_assets.py
│   └── download_testing_data.py
├── third_party/         # Git submodules
│   ├── cld/
│   ├── layerd/
│   ├── llava/
│   └── ultralytics/
├── checkpoints/         # Model weights (not committed)
│   ├── rtdetr/
│   ├── flux/
│   └── cld/
├── data/                # Datasets (not committed)
│   └── dlcv_bbox_dataset/
└── outputs/             # Pipeline outputs (not committed)
    └── pipeline_outputs/
```

---

## Troubleshooting

### Environment Issues

**Q: How to verify environments are set up correctly?**
```bash
conda env list
conda run -n ultralytics python -c "import ultralytics; print(ultralytics.__version__)"
```

**Q: How to recreate environments?**
```bash
python scripts/setup_environments.py --all --force
```

### Training Issues

**Q: GPU OOM during training?**
- Reduce batch size: `batch=8`
- Use disk cache: `cache='disk'`
- Reduce image size: `imgsz=512`

**Q: Training too slow?**
- Enable caching: `cache='disk'`
- Increase workers: `workers=16`
- Use smaller model: `rtdetr-x.pt`

**Q: How to resume training?**
```python
# Load last checkpoint
model = RTDETR("checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/last.pt")
```

### Pipeline Issues

**Q: Pipeline execution failed?**
- Check configuration paths
- Verify input data exists
- Check environment activation
- See [PIPELINE_README.md](PIPELINE_README.md) for detailed troubleshooting

---

## Related Documentation

- [Pipeline Usage Guide](PIPELINE_README.md) - Complete pipeline documentation
- [Environment Setup Guide](scripts/README_SETUP.md) - Detailed environment setup
- [Configuration Examples](configs/exp001/pipeline.yaml) - Pipeline configuration examples

---

## Acknowledgments

- [RT-DETR](https://github.com/ultralytics/ultralytics) - Ultralytics
- [LayerD](https://github.com/CyberAgentAILab/LayerD) - CyberAgent AI Lab
- [CLD](https://github.com/monkek123King/CLD)
- [LLaVA](https://github.com/haotian-liu/LLaVA)

