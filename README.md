# DLCV Final Project - Layout Decomposition Pipeline

本專案實現了一個完整的圖像層級分解 Pipeline，從物件偵測到最終的分層合成。

## 重要連結
- [CLD Repo](https://github.com/monkek123King/CLD/?tab=readme-ov-file)
- [CLD hugging face](https://huggingface.co/thuteam/CLD)
- [LayerD Repo](https://github.com/CyberAgentAILab/LayerD)
- [RTDETR](https://github.com/ultralytics/ultralytics)
- [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#llava-weights)
- [Crello Dataset](https://huggingface.co/datasets/cyberagent/crello)
- [TA Dataset](https://huggingface.co/datasets/WalkerHsu/DLCV2025_final_project_piccollage)

## 📋 目錄

- [專案簡介](#專案簡介)
- [快速開始](#快速開始)
- [專案結構](#專案結構)
- [主要功能](#主要功能)
- [環境設置](#環境設置)
- [Pipeline 使用](#pipeline-使用)
- [模型訓練](#模型訓練)
- [工具與腳本](#工具與腳本)
- [專案結構詳解](#專案結構詳解)

---

## 🎯 專案簡介

本專案是一個端到端的圖像處理 Pipeline，包含以下主要步驟：

1. **RTDETR Detection** - 使用 RT-DETR 模型偵測圖像中的物件
2. **LayerD Decomposition** - 使用 LayerD 模型將圖像分解為多個層級
3. **CLD Format Conversion** - 將偵測結果和層級資訊轉換為 CLD 推理格式
4. **VLM Caption Generation** (可選) - 使用 LLaVA 生成圖像描述
5. **CLD Inference** - 使用 CLD 模型進行最終的分層合成

### 技術棧

- **物件偵測**: RT-DETR (Ultralytics)
- **層級分解**: LayerD (CyberAgent)
- **分層合成**: CLD (Conditional Layout Diffusion)
- **視覺語言模型**: LLaVA (Large Language and Vision Assistant)

---

## 🚀 快速開始

### 1. 克隆專案並初始化 Submodules

```bash
git clone <repository-url>
cd finals_repo
git submodule update --init --recursive
```

### 2. 設置環境

```bash
# 設置所有必要的環境
python scripts/setup_environments.py --all
```

詳細說明請參考：[環境設置指南](scripts/README_SETUP.md)

### 3. 準備配置檔案

```bash
# 複製範例配置
cp configs/exp001/pipeline.yaml configs/my_experiment/pipeline.yaml
# 編輯配置檔案，設定輸入輸出路徑
```

### 5. 執行 Pipeline

```bash
# 執行完整 pipeline
python -m src.pipeline.steps.step_rtdetr --config configs/my_experiment/pipeline.yaml
python -m src.pipeline.steps.step_layerd --config configs/my_experiment/pipeline.yaml
python -m src.pipeline.steps.step_conversion --config configs/my_experiment/pipeline.yaml
python -m src.pipeline.steps.step_vlm --config configs/my_experiment/pipeline.yaml  # 可選
python -m src.pipeline.steps.step_cld --config configs/my_experiment/pipeline.yaml 
```

詳細說明請參考：[Pipeline 使用指南](PIPELINE_README.md)

---

## 📁 專案結構

```
finals_repo/
├── configs/              # 配置檔案
│   └── exp001/          # 實驗配置
│       ├── pipeline.yaml    # Pipeline 主配置
│       └── cld/             # CLD 推理配置
├── src/                  # 原始碼
│   ├── pipeline/        # Pipeline orchestration
│   │   └── steps/      # 各步驟的執行腳本
│   ├── bbox/           # RTDETR 相關
│   ├── layerd/         # LayerD 相關
│   ├── adapters/       # 格式轉換適配器
│   ├── caption/        # VLM caption 生成
│   ├── cld/            # CLD 推理 wrapper
│   └── data/           # Dataset 處理工具
├── scripts/             # 工具腳本
│   ├── setup_environments.py  # 環境設置
│   ├── download_cld_assets.py # CLD 模型下載
│   ├── download_testing_data.py # 測試資料下載
│   └── README_SETUP.md        # 環境設置說明
├── third_party/         # 第三方依賴 (git submodules)
│   ├── cld/            # CLD 模型
│   ├── layerd/         # LayerD 模型
│   ├── llava/          # LLaVA 模型
│   └── ultralytics/    # Ultralytics RT-DETR
├── checkpoints/         # 模型權重 (不 commit)
│   ├── rtdetr/         # RTDETR checkpoints
│   ├── flux/           # FLUX 模型
│   └── cld/            # CLD checkpoints
├── data/                # Dataset (不 commit)
│   └── dlcv_bbox_dataset/  # RTDETR 訓練資料集
├── outputs/             # Pipeline 輸出 (不 commit)
│   └── pipeline_outputs/   # 各步驟的中間產物
└── hpc/                 # HPC/Slurm 腳本
    └── scripts/        # HPC job scripts
```

---

## 🔧 主要功能

### Pipeline 執行

完整的圖像處理流程，從輸入圖像到最終的分層合成結果。

**詳細說明**: [PIPELINE_README.md](PIPELINE_README.md)

### 環境管理

自動設置和管理所有必要的 conda 和 uv 環境。

**詳細說明**: [scripts/README_SETUP.md](scripts/README_SETUP.md)

### 模型訓練

支援 RTDETR 模型的 fine-tuning，針對 layout analysis 任務優化。

**詳細說明**: 見下方 [模型訓練](#模型訓練) 章節

### 工具與可視化

提供多種工具腳本，包括 bbox 可視化、資料集準備等。

**詳細說明**: 見下方 [工具與腳本](#工具與腳本) 章節

---

## 🌍 環境設置

本專案使用多個獨立的環境來管理不同的依賴：

| 環境 | 類型 | 名稱 | 用途 |
|------|------|------|------|
| CLD | Conda | `CLD` | CLD 格式轉換和推理 |
| LayerD | uv | - | LayerD 層級分解 |
| Ultralytics | Conda | `ultralytics` | RTDETR 物件偵測 |
| LLaVA | Conda | `llava` | VLM Caption 生成 |

### 快速設置

```bash
# 設置所有環境
python scripts/setup_environments.py --all

# 或只設置特定環境
python scripts/setup_environments.py --cld --ultralytics --llava --layerd
```

### 詳細說明

完整的環境設置指南、故障排除和驗證方法，請參考：

📖 **[環境設置完整指南](scripts/README_SETUP.md)**

---

## 🔄 Pipeline 使用

### 基本使用

Pipeline 包含 5 個主要步驟，可以逐步執行或一次性執行：

```bash
# Step 1: RTDETR Detection
python -m src.pipeline.steps.step_rtdetr --config configs/exp001/pipeline.yaml

# Step 2: LayerD Decomposition
python -m src.pipeline.steps.step_layerd --config configs/exp001/pipeline.yaml

# Step 3: CLD Format Conversion
python -m src.pipeline.steps.step_conversion --config configs/exp001/pipeline.yaml

# Step 3.5: VLM Caption Generation (可選)
python -m src.pipeline.steps.step_vlm --config configs/exp001/pipeline.yaml

# Step 4: CLD Inference
python -m src.pipeline.steps.step_cld --config configs/exp001/pipeline.yaml
```

### 詳細說明

完整的 Pipeline 使用指南、配置說明、輸出格式和故障排除，請參考：

📖 **[Pipeline 使用完整指南](PIPELINE_README.md)**

---

## 🎓 模型訓練

### RTDETR Fine-tuning

本專案支援針對 layout analysis 任務對 RTDETR 模型進行 fine-tuning。

#### 1. 準備資料集

使用提供的腳本從 HuggingFace 下載並處理資料集：

```bash
# 準備 DLCV Bounding Box Dataset
python -m src.data.dlcv_bbox_dataset

# 預設會下載 20000 張圖片到 data/dlcv_bbox_dataset/
# 可以修改 target_total 參數調整數量
```

**資料集來源**: `WalkerHsu/DLCV2025_final_project_piccollage` (HuggingFace)

**資料集處理邏輯**:
- 從 PicCollage 資料集中提取 layout elements
- 處理旋轉物件的幾何變換（AABB 計算）
- 對於無旋轉物件，使用 Alpha Crop 獲得更緊密的 bounding box
- 自動過濾背景層（>95% canvas 面積）
- 轉換為 YOLO 格式（normalized coordinates）

**輸出結構**:
```
data/dlcv_bbox_dataset/
├── data.yaml          # YOLO 資料集配置
├── images/
│   ├── train/         # 訓練圖片 (90%)
│   └── val/           # 驗證圖片 (10%)
└── labels/
    ├── train/         # 訓練標籤
    └── val/           # 驗證標籤
```

#### 2. 訓練模型

```bash
# 使用 conda ultralytics 環境
conda activate ultralytics

# 執行訓練
python -m src.bbox.train_rtdetr

# 或直接使用 conda run
conda run -n ultralytics python -m src.bbox.train_rtdetr
```

**訓練配置**:
- **模型**: RTDETR-L (Large)
- **Epochs**: 100
- **Batch Size**: 16 (V100 GPU，如 OOM 可降至 8)
- **Image Size**: 640x640
- **Optimizer**: AdamW
- **Learning Rate**: 0.0001
- **特殊設定**:
  - 關閉 Mosaic 和 Mixup（避免破壞 layout 邏輯）
  - 關閉旋轉增強（layout 通常是直立的）
  - 保留安全的增強（縮放、翻轉、顏色變化）

**輸出位置**: `checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt`

#### 3. 使用訓練好的模型

在 `pipeline.yaml` 中指定模型路徑：

```yaml
rtdetr:
  model_path: "checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt"
```

---

## 🛠️ 工具與腳本

### 環境設置

- **`scripts/setup_environments.py`** - Python 版本的環境設置腳本
- **`scripts/setup_environments.sh`** - Shell 版本的環境設置腳本

詳細說明: [scripts/README_SETUP.md](scripts/README_SETUP.md)

### RTDETR Checkpoint 下載

使用 `gdown` 從 Google Drive 下載訓練好的 RTDETR checkpoint：

```bash
# 安裝 gdown（如果尚未安裝）
pip install gdown

# 創建目錄（如果不存在）
mkdir -p checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights

# 下載 RTDETR checkpoint
gdown --id 1TT5iBr1ber8pT0E7tcfUE-FV1ssn4dcQ -O checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt
```

**下載後**：
- Checkpoint 會保存到 `checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt`
- 這是 pipeline 配置中的預設路徑，無需額外配置即可使用

**驗證下載**：
```bash
# 檢查文件是否存在
ls -lh checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt

# 應該看到文件大小約為數百 MB
```

### CLD 模型下載

- **`scripts/download_cld_assets.py`** - 下載 CLD 所需的模型和權重

**設置 HuggingFace Token**：

某些模型（如 FLUX.1-dev）需要 HuggingFace token 才能下載。你可以通過以下方式設置：

1. **使用環境變數**（推薦）：
```bash
# 設置環境變數
export HF_TOKEN="your_huggingface_token_here"
# 或
export HUGGINGFACE_HUB_TOKEN="your_huggingface_token_here"

# 然後執行下載
python scripts/download_cld_assets.py
```

2. **使用命令行參數**：
```bash
python scripts/download_cld_assets.py --hf-token "your_huggingface_token_here"
```

3. **永久設置**（在 `~/.bashrc` 或 `~/.zshrc` 中）：
```bash
echo 'export HF_TOKEN="your_huggingface_token_here"' >> ~/.bashrc
source ~/.bashrc
```

**獲取 HuggingFace Token**：
1. 前往 [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. 創建新的 token（需要 `read` 權限）
3. 複製 token 並使用上述方式設置

**執行下載**：
```bash
python scripts/download_cld_assets.py
```

下載內容包括：
- FLUX.1-dev 模型
- ControlNet Inpainting Alpha adapter
- CLD LoRA 權重
- Transparent VAE 權重

**注意**：如果沒有設置 token，腳本會顯示警告，某些需要授權的模型可能無法下載。

### 測試資料下載

- **`scripts/download_testing_data.py`** - 從 Google Drive 下載測試資料並解壓縮

**使用方法**：

1. **在腳本中設置 File ID**（推薦）：
   
   編輯 `scripts/download_testing_data.py`，在第 27 行填寫 Google Drive file ID：
   ```python
   DEFAULT_FILE_ID = "YOUR_FILE_ID_HERE"  # 填寫您的 Google Drive file ID
   ```
   
   然後直接運行：
   ```bash
   python scripts/download_testing_data.py
   ```

2. **使用命令行參數**：
   ```bash
   # 使用 file ID
   python scripts/download_testing_data.py --file-id "YOUR_FILE_ID"
   
   # 使用完整 URL
   python scripts/download_testing_data.py --url "https://drive.google.com/uc?id=YOUR_FILE_ID"
   
   # 指定輸出目錄
   python scripts/download_testing_data.py --file-id "YOUR_FILE_ID" --output-dir data/test
   
   # 保留下載的壓縮文件
   python scripts/download_testing_data.py --file-id "YOUR_FILE_ID" --keep-archive
   ```

**功能**：
- 自動安裝 `gdown`（如果未安裝）
- 支援多種壓縮格式：`.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tar.xz`
- 自動解壓縮到 `data/` 目錄（或指定的輸出目錄）
- 預設會在下載後刪除壓縮文件（可使用 `--keep-archive` 保留）

**獲取 Google Drive File ID**：
- 從分享連結中提取：`https://drive.google.com/file/d/FILE_ID_HERE/view`
- 或從直接下載連結：`https://drive.google.com/uc?id=FILE_ID_HERE`

### 資料集準備

- **`src/data/dlcv_bbox_dataset.py`** - 準備 RTDETR 訓練資料集

```bash
python -m src.data.dlcv_bbox_dataset
```

### 可視化工具

- **`src/bbox/visualize_bbox_gif.py`** - 將 CLD JSON 中的 bbox 可視化為 GIF

```bash
# 單個檔案
python -m src.bbox.visualize_bbox_gif \
  --input outputs/pipeline_outputs/cld/image1.json \
  --output outputs/pipeline_outputs/cld/image1.gif \
  --use-quantized

# 整個目錄
python -m src.bbox.visualize_bbox_gif \
  --input outputs/pipeline_outputs/cld \
  --output-dir outputs/pipeline_outputs/cld_gif \
  --use-quantized
```

**功能**:
- 從 CLD JSON 檔案讀取 bbox 資訊
- 生成逐層顯示的 GIF 動畫
- 支援 `ordered_bboxes` 或 `quantized_boxes`
- 每個 frame 聚焦當前 bbox，之前的 bbox 以半透明灰色顯示

---

## 📂 專案結構詳解

### `src/` - 原始碼

- **`src/pipeline/`** - Pipeline orchestration
  - `steps/` - 各步驟的執行腳本（step_rtdetr.py, step_layerd.py, 等）
  
- **`src/bbox/`** - RTDETR 相關
  - `infer.py` - RTDETR 推理
  - `train_rtdetr.py` - RTDETR 訓練
  - `visualize_bbox_gif.py` - Bbox 可視化工具
  
- **`src/layerd/`** - LayerD 相關
  - `infer.py` - LayerD 推理和 mask 提取
  
- **`src/adapters/`** - 格式轉換
  - `rtdetr_layerd_to_cld_infer.py` - 將 RTDETR + LayerD 結果轉換為 CLD 格式
  
- **`src/caption/`** - VLM Caption
  - `generate.py` - 使用 LLaVA 生成 captions
  
- **`src/cld/`** - CLD 推理
  - `infer_dlcv.py` - CLD 推理 wrapper
  
- **`src/data/`** - 資料集處理
  - `dlcv_bbox_dataset.py` - DLCV 資料集準備
  - `custom_cld_dataset.py` - CLD 自定義資料集

### `configs/` - 配置檔案

- **`configs/exp001/pipeline.yaml`** - Pipeline 主配置
- **`configs/exp001/cld/infer.yaml`** - CLD 推理配置

### `scripts/` - 工具腳本

- **`setup_environments.py`** - 環境設置（Python）
- **`setup_environments.sh`** - 環境設置（Shell）
- **`download_cld_assets.py`** - CLD 模型下載
- **`download_testing_data.py`** - 測試資料下載（從 Google Drive）

### `third_party/` - 第三方依賴

所有第三方依賴都使用 git submodule 管理：
- `cld/` - CLD 模型
- `layerd/` - LayerD 模型
- `llava/` - LLaVA 模型
- `ultralytics/` - Ultralytics RT-DETR

### `checkpoints/` - 模型權重

- `rtdetr/` - RTDETR checkpoints
- `flux/` - FLUX 模型
- `cld/` - CLD checkpoints

**注意**: 此目錄不應 commit 到 git，請確保在 `.gitignore` 中。

### `data/` - 資料集

- `dlcv_bbox_dataset/` - RTDETR 訓練資料集

**注意**: 此目錄不應 commit 到 git。

### `outputs/` - Pipeline 輸出

- `pipeline_outputs/` - 各步驟的中間產物和最終結果
  - `rtdetr/` - RTDETR 輸出 (JSON)
  - `layerd/` - LayerD 輸出 (NPZ)
  - `cld/` - CLD 格式輸出 (JSON)
  - `cld_inference/` - CLD 最終推理結果

**注意**: 此目錄不應 commit 到 git。

---

## 📚 相關文檔

- **[Pipeline 使用指南](PIPELINE_README.md)** - 完整的 Pipeline 使用說明
- **[環境設置指南](scripts/README_SETUP.md)** - 環境設置和故障排除
- **[配置檔案範例](configs/exp001/pipeline.yaml)** - Pipeline 配置範例
- **[CLD Inference 配置範例](configs/exp001/cld/infer.yaml)** - CLD 推理配置範例

---

## 🔍 常見問題

### 環境問題

**Q: 如何檢查環境是否正確設置？**

```bash
# 檢查 conda 環境
conda env list

# 測試各環境
conda run -n CLD python --version
conda run -n ultralytics python -c "import ultralytics; print(ultralytics.__version__)"
conda run -n llava python --version

# 測試 LayerD (需要 cd 到目錄)
cd third_party/layerd
uv run python --version
```

**Q: 如何重新設置環境？**

```bash
# 強制重新創建所有環境
python scripts/setup_environments.py --all --force
```

### Pipeline 問題

**Q: Pipeline 執行失敗怎麼辦？**

請參考 [PIPELINE_README.md](PIPELINE_README.md) 中的 [故障排除](PIPELINE_README.md#故障排除) 章節。

**Q: 如何只執行部分步驟？**

每個步驟都可以獨立執行，只需要確保前置步驟的輸出存在。參考 [PIPELINE_README.md](PIPELINE_README.md) 中的 [執行方式](PIPELINE_README.md#執行方式) 章節。

### 訓練問題

**Q: RTDETR 訓練時 GPU OOM 怎麼辦？**

在 `src/bbox/train_rtdetr.py` 中調整 `batch` 參數（例如從 16 改為 8）。

**Q: 如何下載更多訓練資料？**

修改 `src/data/dlcv_bbox_dataset.py` 中的 `target_total` 參數。

---

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

---

## 📖 RTDETR Fine-tuning 完整指南

本指南詳細說明如何針對 Layout Analysis 任務對 RTDETR 模型進行 fine-tuning。

### 📋 目錄

- [環境準備](#環境準備)
- [資料集準備](#資料集準備)
- [訓練模型](#訓練模型)
- [訓練配置詳解](#訓練配置詳解)
- [監控訓練過程](#監控訓練過程)
- [使用訓練好的模型](#使用訓練好的模型)
- [常見問題與故障排除](#常見問題與故障排除)

---

### 🔧 環境準備

#### 1. 確保 Ultralytics 環境已設置

```bash
# 檢查環境是否存在
conda env list | grep ultralytics

# 如果不存在，執行環境設置
python scripts/setup_environments.py --ultralytics
```

#### 2. 驗證環境

```bash
# 激活環境
conda activate ultralytics

# 驗證 Ultralytics 版本
python -c "import ultralytics; print(ultralytics.__version__)"

# 驗證 CUDA 可用性（訓練需要 GPU）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### 📦 資料集準備

#### 1. 準備 DLCV Bounding Box Dataset

```bash
# 激活環境
conda activate ultralytics

# 執行資料集準備腳本
python -m src.data.dlcv_bbox_dataset
```

**預設設定**：
- 下載 20,000 張圖片
- 自動劃分 train/val（90%/10%）
- 輸出位置：`data/dlcv_bbox_dataset/`

**自訂資料量**：

編輯 `src/data/dlcv_bbox_dataset.py`，修改最後一行：

```python
if __name__ == "__main__":
    prepare_dlcv_bbox_dataset(target_total=50000)  # 改為你想要的數量
```

#### 2. 資料集結構

```
data/dlcv_bbox_dataset/
├── data.yaml              # YOLO 資料集配置
├── images/
│   ├── train/            # 訓練圖片 (90%)
│   └── val/              # 驗證圖片 (10%)
└── labels/
    ├── train/            # 訓練標籤 (YOLO 格式)
    └── val/              # 驗證標籤 (YOLO 格式)
```

#### 3. 資料集類別

資料集包含兩個類別：

- **Class 0: `layout_element`** - 一般布局元素（圖片、圖形等）
- **Class 1: `text`** - 文字元素

`data.yaml` 內容範例：

```yaml
path: /absolute/path/to/data/dlcv_bbox_dataset
train: images/train
val: images/val
names:
  0: layout_element
  1: text
```

#### 4. 資料集處理邏輯

資料集準備腳本會自動處理：

- ✅ **旋轉物件**：使用幾何變換計算 AABB（Axis-Aligned Bounding Box）
- ✅ **無旋轉物件**：使用 Alpha Crop 獲得更緊密的 bounding box
- ✅ **背景過濾**：自動過濾佔 canvas 面積 >95% 的背景層
- ✅ **類別標註**：根據 element type 自動標註為 layout_element 或 text
- ✅ **格式轉換**：轉換為 YOLO 格式（normalized coordinates）

#### 5. 驗證資料集

```bash
# 檢查資料集結構
ls -lh data/dlcv_bbox_dataset/images/train/ | head -5
ls -lh data/dlcv_bbox_dataset/labels/train/ | head -5

# 檢查標籤格式（應該看到 class_id x_center y_center width height）
head -3 data/dlcv_bbox_dataset/labels/train/*.txt
```

---

### 🚀 訓練模型

#### 1. 基本訓練命令

```bash
# 激活環境
conda activate ultralytics

# 執行訓練
python -m src.bbox.train_rtdetr
```

或使用 `conda run`（無需手動激活）：

```bash
conda run -n ultralytics python -m src.bbox.train_rtdetr
```

#### 2. 訓練輸出

訓練過程會自動：

- 下載預訓練模型（如果不存在）：`checkpoints/rtdetr/rtdetr-l.pt`
- 保存訓練日誌：`checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/`
- 保存最佳模型：`checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt`
- 保存最後模型：`checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/last.pt`

#### 3. 訓練時間估算

- **20K 圖片，100 epochs，batch=16，V100 GPU**：約 8-12 小時
- **50K 圖片，100 epochs，batch=16，V100 GPU**：約 20-30 小時

---

### ⚙️ 訓練配置詳解

#### 模型設定

- **預訓練模型**：RTDETR-L (Large)
  - 平衡準確度和速度的最佳選擇
  - 自動下載到 `checkpoints/rtdetr/rtdetr-l.pt`

#### 訓練超參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `epochs` | 100 | 訓練輪數（足夠 fine-tuning） |
| `patience` | 15 | Early stopping：15 epochs 無改善則停止 |
| `batch` | 16 | Batch size（V100 GPU，OOM 可降至 8） |
| `imgsz` | 640 | 輸入圖像尺寸 |
| `optimizer` | AdamW | Transformer 模型推薦優化器 |
| `lr0` | 0.0001 | 初始學習率（fine-tuning 使用較小值） |
| `workers` | 8 | 數據加載線程數 |
| `cache` | True | 緩存圖像到 RAM，加速訓練 |
| `amp` | True | 自動混合精度（節省記憶體） |

#### 數據增強策略

**關閉的增強**（避免破壞 layout 邏輯）：

- ❌ **Mosaic** (`mosaic=0.0`)：避免破壞布局邏輯
- ❌ **Mixup** (`mixup=0.0`)：避免透明度混淆
- ❌ **旋轉** (`degrees=0.0`)：Layout 通常是直立的

**保留的增強**（安全的增強）：

- ✅ **縮放** (`scale=0.5`)：隨機縮放 ±50%，適應不同 canvas 尺寸
- ✅ **水平翻轉** (`fliplr=0.5`)：Layout 通常左右對稱
- ✅ **顏色變化** (`hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`)：色調、飽和度、亮度變化

#### 自訂訓練配置

如需修改訓練參數，編輯 `src/bbox/train_rtdetr.py`：

```python
results = model.train(
    data=DATASET_PATH / "data.yaml",
    epochs=150,        # 增加訓練輪數
    batch=8,           # 降低 batch size（如果 OOM）
    lr0=0.00005,       # 降低學習率
    # ... 其他參數
)
```

---

### 📊 監控訓練過程

#### 1. 訓練日誌

訓練過程會顯示：

- 當前 epoch 和總 epochs
- 訓練和驗證 loss
- mAP (mean Average Precision) 指標
- 訓練速度（images/sec）

#### 2. TensorBoard（可選）

Ultralytics 會自動記錄訓練日誌，可以使用 TensorBoard 可視化：

```bash
# 安裝 TensorBoard（如果尚未安裝）
pip install tensorboard

# 啟動 TensorBoard
tensorboard --logdir checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/

# 在瀏覽器打開 http://localhost:6006
```

#### 3. 檢查訓練結果

```bash
# 查看訓練目錄
ls -lh checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/

# 查看最佳模型
ls -lh checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt

# 查看訓練曲線（如果生成了 results.png）
open checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/results.png
```

---

### 🎯 使用訓練好的模型

#### 1. 在 Pipeline 中使用

編輯 `configs/exp001/pipeline.yaml`（或你的配置檔案）：

```yaml
rtdetr:
  model_path: "checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt"
  conf_threshold: 0.25
  iou_threshold: 0.45
```

#### 2. 直接使用模型進行推理

```python
from ultralytics import RTDETR

# 載入訓練好的模型
model = RTDETR("checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt")

# 進行推理
results = model("path/to/image.jpg")

# 可視化結果
results[0].show()
```

#### 3. 驗證模型性能

```bash
# 使用驗證集評估模型
python -c "
from ultralytics import RTDETR
model = RTDETR('checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt')
metrics = model.val(data='data/dlcv_bbox_dataset/data.yaml')
print(f'mAP50: {metrics.box.map50:.4f}')
print(f'mAP50-95: {metrics.box.map:.4f}')
"
```

---

### ❓ 常見問題與故障排除

#### Q1: GPU 記憶體不足 (OOM)

**解決方案**：

1. **降低 batch size**：
   ```python
   batch=8  # 從 16 改為 8
   ```

2. **關閉圖像緩存**：
   ```python
   cache=False  # 從 True 改為 False
   ```

3. **降低圖像尺寸**：
   ```python
   imgsz=512  # 從 640 改為 512（可能影響準確度）
   ```

#### Q2: 訓練速度太慢

**解決方案**：

1. **啟用圖像緩存**：
   ```python
   cache=True  # 確保啟用
   ```

2. **增加 workers**：
   ```python
   workers=16  # 根據 CPU 核心數調整
   ```

3. **使用更小的模型**：
   ```python
   model = RTDETR("rtdetr-x.pt")  # 使用 Extra Small 版本（更快但準確度較低）
   ```

#### Q3: 驗證 loss 不下降或過擬合

**解決方案**：

1. **降低學習率**：
   ```python
   lr0=0.00005  # 從 0.0001 降低
   ```

2. **增加 Early Stopping patience**：
   ```python
   patience=20  # 從 15 增加
   ```

3. **增加數據增強**（如果尚未啟用）：
   ```python
   scale=0.5
   fliplr=0.5
   ```

#### Q4: 模型準確度不理想

**解決方案**：

1. **增加訓練資料量**：
   ```python
   prepare_dlcv_bbox_dataset(target_total=50000)  # 增加資料量
   ```

2. **增加訓練輪數**：
   ```python
   epochs=150  # 從 100 增加
   ```

3. **使用更大的模型**：
   ```python
   model = RTDETR("rtdetr-x.pt")  # 使用 Extra Large 版本
   ```

#### Q5: 資料集下載失敗

**解決方案**：

1. **檢查網路連接**：
   ```bash
   # 測試 HuggingFace 連接
   python -c "from datasets import load_dataset; print('OK')"
   ```

2. **使用代理或 VPN**（如果在某些地區）

3. **手動下載資料集**：
   ```python
   # 在 Python 中手動下載
   from datasets import load_dataset
   dataset = load_dataset("WalkerHsu/DLCV2025_final_project_piccollage", split="train")
   ```

#### Q6: 訓練中斷如何恢復？

**解決方案**：

Ultralytics 會自動保存 `last.pt`，可以從中恢復：

```python
# 修改 train_rtdetr.py，載入 last.pt 而不是預訓練模型
model = RTDETR("checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/last.pt")
```

---

### 📝 訓練檢查清單

在開始訓練前，確認：

- [ ] Ultralytics 環境已正確設置
- [ ] GPU 可用且 CUDA 正常
- [ ] 資料集已準備完成（檢查 `data/dlcv_bbox_dataset/`）
- [ ] `data.yaml` 配置正確（包含兩個類別）
- [ ] 有足夠的磁碟空間（至少 10GB 用於 checkpoints）
- [ ] 有足夠的 GPU 記憶體（V100 16GB 推薦 batch=16）

---

### 🔗 相關資源

- [Ultralytics RTDETR 文檔](https://docs.ultralytics.com/models/rtdetr/)
- [YOLO 格式說明](https://docs.ultralytics.com/datasets/)
- [訓練最佳實踐](https://docs.ultralytics.com/modes/train/)

---

## 📄 授權

[請根據實際情況填寫授權資訊]

---

## 🙏 致謝

- [RT-DETR](https://github.com/ultralytics/ultralytics) - Ultralytics
- [LayerD](https://github.com/CyberAgentAILab/LayerD) - CyberAgent AI Lab
- [CLD](https://github.com/monkek123King/CLD)
- [LLaVA](https://github.com/haotian-liu/LLaVA)

