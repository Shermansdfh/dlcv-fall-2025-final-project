# Depth Experiments Guide

本指南說明如何使用 `depth_exp` 目錄中的工具進行深度相關的實驗。

## 📁 目錄結構

```
depth_exp/
├── GUIDE.md                    # 本指南
├── caption_llava15.json         # LLaVA 生成的圖像 caption
├── CLD/                         # Controllable Layer Decomposition
│   ├── train/                   # 訓練相關
│   │   ├── train.py             # 原始 CLD 訓練腳本
│   │   ├── train.yaml           # 原始訓練配置
│   │   ├── train_dlcv.py        # DLCV 數據集訓練腳本（僅訓練 MLCA）
│   │   └── train_dlcv.yaml     # DLCV 訓練配置
│   ├── infer/                   # 推理相關
│   ├── eval/                    # 評估相關
│   └── models/                  # 模型定義
└── ml-depth-pro/                # Depth Pro 深度估計模型
    ├── src/depth_pro/           # Depth Pro 核心代碼
    └── get_pretrained_models.sh # 下載預訓練模型
```

## 🚀 快速開始

### 1. 環境設置

#### CLD 環境
```bash
cd depth_exp/CLD
conda env create -f environment.yml
conda activate CLD
```

#### Depth Pro 環境（可選，用於深度 channel）
```bash
cd depth_exp/ml-depth-pro
conda create -n depth-pro -y python=3.9
conda activate depth-pro
pip install -e .
source get_pretrained_models.sh  # 下載預訓練模型
```

### 2. 準備模型權重

#### 下載 FLUX.1-dev 模型
```python
from huggingface_hub import snapshot_download

repo_id = "black-forest-labs/FLUX.1-dev"
snapshot_download(repo_id, local_dir="path/to/FLUX.1-dev")
```

#### 下載 Adapter 權重
```python
repo_id = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha"
snapshot_download(repo_id, local_dir="path/to/adapter")
```

#### 下載 CLD LoRA 權重（從 HuggingFace）
訪問 https://huggingface.co/thuteam/CLD 下載以下文件：
```
ckpt/
├── decouple_LoRA/
│   ├── adapter/
│   │   └── pytorch_lora_weights.safetensors
│   ├── layer_pe.pth
│   └── transformer/
│       └── pytorch_lora_weights.safetensors
├── pre_trained_LoRA/
│   └── pytorch_lora_weights.safetensors
└── prism_ft_LoRA/
    └── pytorch_lora_weights.safetensors
```

## 📊 使用 DLCV 數據集訓練 CLD

### 概述

`train_dlcv.py` 是專門為 DLCV 數據集設計的訓練腳本，具有以下特點：

- ✅ 使用 `DLCVCLDDataset` 從 HuggingFace 載入數據
- ✅ 自動從 `caption_llava15.json` 讀取 caption
- ✅ 可選啟用深度 channel（使用 ml-depth-pro）
- ✅ **僅訓練 MultiLayer-Adapter (MLCA)**，Transformer 完全凍結

### 配置訓練

編輯 `depth_exp/CLD/train/train_dlcv.yaml`：

```yaml
# 基本配置
seed: 42
max_layer_num: 52
max_steps: 200000
log_every: 1000
save_every: 1000
accum_steps: 4

# LoRA 配置
lora_rank: 64
lora_alpha: 64
lora_dropout: 0

# 數據集配置
train_max_samples: null  # null 表示使用所有樣本，或指定數量如 20000
dataset_seed: 42
shuffle_buffer_size: 2000
caption_json_path: null  # null 使用默認路徑 (depth_exp/caption_llava15.json)

# 深度 channel（可選）
use_depth: false  # 設為 true 啟用深度 channel
depth_device: null  # null 自動選擇 (cuda/cpu)

# 模型路徑
pretrained_model_name_or_path: "path/to/FLUX.1-dev"
pretrained_adapter_path: "path/to/adapter"
pretrained_lora_dir: "path/to/pre_trained_LoRA"  # 可選
artplus_lora_dir: null  # 可選

# 輸出
output_dir: "path/to/save/checkpoints"
resume_from: null  # 可選：從 checkpoint 恢復訓練
```

### 開始訓練

```bash
cd depth_exp/CLD/train
conda activate CLD
python train_dlcv.py -c train_dlcv.yaml
```

### 訓練選項說明

#### 1. 基本訓練（不使用深度）
```yaml
use_depth: false
train_max_samples: 20000  # 限制樣本數量
```

#### 2. 使用深度 channel 訓練
```yaml
use_depth: true
depth_device: "cuda"  # 或 "cpu"
```

**注意**：使用深度 channel 需要：
- 已安裝 ml-depth-pro
- 已下載 Depth Pro 預訓練模型
- 更多 GPU 記憶體

#### 3. 從 checkpoint 恢復
```yaml
resume_from: "path/to/checkpoint/directory"
```

Checkpoint 目錄結構：
```
checkpoint_dir/
├── adapter/
│   ├── pytorch_lora_weights.safetensors
│   ├── optimizer.bin
│   └── scheduler.bin
├── transformer/  # 空目錄（MLCA-only 訓練）
└── layer_pe.pth
```

### 監控訓練

訓練過程中會：
- 在 `output_dir` 保存 checkpoint
- 在 TensorBoard 記錄 loss（`tensorboard --logdir output_dir`）
- 在終端顯示進度條和 loss

## 🔍 數據集說明

### DLCVCLDDataset

位置：`src/data/dlcv_cld_dataset.py`

**功能**：
- 從 HuggingFace 載入 `WalkerHsu/DLCV2025_final_project_piccollage` 數據集
- 自動處理圖像層（layers）和邊界框（bounding boxes）
- 支援旋轉和 alpha crop 處理
- 從 `caption_llava15.json` 讀取 caption
- 可選添加深度 channel

**數據格式**：
- `pixel_RGBA`: 每個 layer 的 RGBA tensor 列表
- `pixel_RGB`: 每個 layer 的 RGB tensor 列表
- `whole_img`: 完整圖像的 RGB PIL Image
- `caption`: 文字描述
- `layout`: 邊界框列表 `[[x1, y1, x2, y2], ...]`

### Caption 文件

`caption_llava15.json` 格式：
```json
{
  "/path/to/image/00000000.png": "Caption text here...",
  "/path/to/image/00000001.png": "Another caption...",
  ...
}
```

Dataset 會根據圖像 ID 自動匹配 caption。

## 🎯 訓練策略

### MLCA-Only 訓練

`train_dlcv.py` 專門設計為**僅訓練 MultiLayer-Adapter**：

- ✅ Transformer 完全凍結（`requires_grad=False`, `eval()` 模式）
- ✅ 僅訓練 Adapter 的 LoRA 權重和 layer_pe
- ✅ 更快的訓練速度
- ✅ 更少的記憶體使用
- ✅ 適合在 DLCV 數據集上微調

### 與原始訓練的區別

| 特性 | `train.py` (原始) | `train_dlcv.py` (DLCV) |
|------|------------------|------------------------|
| 數據集 | PrismLayersPro | DLCV (HuggingFace) |
| Caption | 數據集內建 | 從 JSON 文件讀取 |
| 深度 channel | ❌ | ✅ 可選 |
| 訓練目標 | Transformer + MLCA | 僅 MLCA |
| Transformer 狀態 | 可訓練 | 完全凍結 |

## 🔧 故障排除

### 問題 1: 找不到 depth_pro 模組

**錯誤**：
```
ImportError: depth_pro is not available
```

**解決**：
```bash
cd depth_exp/ml-depth-pro
pip install -e .
source get_pretrained_models.sh
```

或設置 `use_depth: false` 不使用深度 channel。

### 問題 2: CUDA 記憶體不足

**解決方案**：
1. 減少 `train_max_samples`
2. 增加 `accum_steps`（梯度累積）
3. 設置 `use_depth: false`
4. 使用更小的 `lora_rank`

### 問題 3: Caption 找不到

**檢查**：
- `caption_llava15.json` 是否存在於 `depth_exp/` 目錄
- 圖像 ID 格式是否匹配（dataset 會自動處理不同格式）

### 問題 4: 數據集載入緩慢

**優化**：
- 設置 `train_max_samples` 限制樣本數量
- 調整 `shuffle_buffer_size`
- 使用本地緩存的 HuggingFace 數據集

## 📈 評估訓練結果

### 檢查 Checkpoint

訓練後，checkpoint 保存在 `output_dir`：
```
output_dir/
├── adapter/
│   ├── pytorch_lora_weights.safetensors  # MLCA LoRA 權重
│   ├── optimizer.bin
│   └── scheduler.bin
├── transformer/  # 空目錄（MLCA-only）
├── layer_pe.pth  # Layer positional encoding
└── random_states_0.pkl  # RNG 狀態
```

### 使用訓練好的模型

在 `infer/infer.yaml` 中設置：
```yaml
adapter_lora_dir: "path/to/output_dir/adapter"
layer_ckpt: "path/to/output_dir"
```

然後運行推理：
```bash
cd depth_exp/CLD/infer
python infer.py -c infer.yaml
```

## 📝 最佳實踐

1. **開始時使用小樣本**：
   ```yaml
   train_max_samples: 1000
   max_steps: 5000
   ```

2. **逐步增加規模**：
   - 先用小樣本驗證流程
   - 確認 loss 正常下降
   - 再使用完整數據集

3. **監控資源使用**：
   - 使用 `nvidia-smi` 監控 GPU
   - 使用 TensorBoard 監控 loss

4. **定期保存**：
   - 設置合理的 `save_every`
   - 重要 checkpoint 手動備份

5. **實驗記錄**：
   - 記錄使用的配置參數
   - 記錄訓練過程中的觀察
   - 保存重要的實驗結果

## 🔗 相關資源

- **CLD 論文**: https://arxiv.org/abs/2511.16249
- **CLD HuggingFace**: https://huggingface.co/thuteam/CLD
- **Depth Pro 論文**: https://arxiv.org/abs/2410.02073
- **FLUX.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev

## 💡 進階用法

### 自定義 Caption 路徑

```yaml
caption_json_path: "path/to/custom_caption.json"
```

### 混合使用深度和原始數據

可以分別訓練兩個模型：
1. 不使用深度的模型（`use_depth: false`）
2. 使用深度的模型（`use_depth: true`）

然後比較結果。

### 調整 LoRA 參數

根據數據集大小調整：
```yaml
# 小數據集
lora_rank: 32
lora_alpha: 32

# 大數據集
lora_rank: 128
lora_alpha: 128
```

## ❓ 常見問題

**Q: 為什麼只訓練 MLCA？**  
A: MLCA 是控制層分解的核心組件，在 DLCV 數據集上微調 MLCA 通常足夠，且訓練更快、更穩定。

**Q: 可以使用原始 `train.py` 訓練 DLCV 數據集嗎？**  
A: 可以，但需要修改 dataset 導入。`train_dlcv.py` 已經整合了 DLCV 數據集和相關功能。

**Q: 深度 channel 是必需的嗎？**  
A: 不是。深度 channel 是可選功能，可以幫助模型理解場景深度，但標準訓練不需要。

**Q: 如何知道訓練是否正常？**  
A: 觀察：
- Loss 應該逐漸下降
- TensorBoard 曲線應該平滑
- 沒有 CUDA 錯誤或記憶體問題

---

**最後更新**: 2025-01-XX  
**維護者**: DLCV Final Project Team

