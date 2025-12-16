# Pipeline 使用指南

本 Pipeline 將圖像處理流程串接為四個主要步驟，從物件偵測到最終的層級合成。

## 📋 目錄

- [快速開始](#快速開始)
- [環境設置](#環境設置)
- [Pipeline 架構](#pipeline-架構)
- [配置檔案](#配置檔案)
- [執行方式](#執行方式)
- [輸出格式](#輸出格式)
- [故障排除](#故障排除)

---

## 🚀 快速開始

### 1. 環境設置

首先設置所有必要的環境：

```bash
# 設置所有環境
python scripts/setup_environments.py --all

# 或只設置特定環境
python scripts/setup_environments.py --cld --ultralytics --llava --layerd
```

詳細說明請參考 [`scripts/README_SETUP.md`](scripts/README_SETUP.md)。

### 2. 準備配置檔案

複製並修改配置檔案：

```bash
cp configs/exp001/pipeline.yaml configs/my_experiment/pipeline.yaml
# 編輯 configs/my_experiment/pipeline.yaml，設定你的輸入輸出路徑
```

### 3. 執行完整 Pipeline

```bash
# 從專案根目錄執行
python -m src.pipeline.steps.step_rtdetr --config configs/my_experiment/pipeline.yaml
python -m src.pipeline.steps.step_layerd --config configs/my_experiment/pipeline.yaml
python -m src.pipeline.steps.step_conversion --config configs/my_experiment/pipeline.yaml
python -m src.pipeline.steps.step_vlm --config configs/my_experiment/pipeline.yaml  # 可選
python -m src.pipeline.steps.step_cld --config configs/my_experiment/pipeline.yaml
```

---

## 🔧 環境設置

### 環境需求

| 環境 | 類型 | 名稱 | 用途 |
|------|------|------|------|
| CLD | Conda | `CLD` | Step 3 (格式轉換) 和 Step 4 (推理) |
| LayerD | uv | - | Step 2 (層分解) |
| Ultralytics | Conda | `ultralytics` | Step 1 (物件偵測) |
| LLaVA | Conda | `llava` | Step 3.5 (Caption 生成，可選) |

### 自動設置

使用提供的腳本自動設置所有環境：

```bash
# 設置所有環境
python scripts/setup_environments.py --all

# 或使用 Shell 腳本
bash scripts/setup_environments.sh --all
```

### 手動設置

如果需要手動設置，請參考 [`scripts/README_SETUP.md`](scripts/README_SETUP.md)。

---

## 🏗️ Pipeline 架構

### Pipeline 流程

```
輸入圖像
    ↓
[Step 1] RTDETR Detection (物件偵測)
    ↓ 輸出: *.json (bboxes)
[Step 2] LayerD Decomposition (層分解)
    ↓ 輸出: *.npz (masks)
[Step 3] CLD Format Conversion (格式轉換)
    ↓ 輸出: *.json (CLD 格式)
[Step 3.5] VLM Caption Generation (可選)
    ↓ 更新: *.json (添加 whole_caption)
[Step 4] CLD Inference (最終推理)
    ↓ 輸出: 合成圖像 (RGBA layers)
```

### 步驟說明

1. **Step 1: RTDETR Detection**
   - 使用 RT-DETR 模型偵測圖像中的物件
   - 輸出每個物件的 bounding box 和信心度

2. **Step 2: LayerD Decomposition**
   - 使用 LayerD 模型將圖像分解為多個層級
   - 輸出前景和背景的 masks

3. **Step 3: CLD Format Conversion**
   - 將 RTDETR 和 LayerD 的結果合併
   - 轉換為 CLD 推理所需的格式
   - 包含層級排序和 box 量化

4. **Step 3.5: VLM Caption Generation** (可選)
   - 使用 LLaVA 模型生成圖像描述
   - 更新 CLD JSON 檔案中的 `whole_caption` 欄位

5. **Step 4: CLD Inference**
   - 使用 CLD 模型進行最終推理
   - 生成分層的 RGBA 圖像

---

## ⚙️ 配置檔案

### Pipeline 配置 (`configs/exp001/pipeline.yaml`)

```yaml
# RTDETR Detection Step
rtdetr:
  input_dir: "inputs"  # 輸入圖片目錄（相對於 config 檔案位置）
  output_dir: "outputs/pipeline_outputs/rtdetr"  # RTDETR 結果輸出目錄
  model_path: "checkpoints/rtdetr/rtdetr_dlcv_bbox_dataset/weights/best.pt"  # 模型路徑
  conf: 0.4  # 信心閾值
  limit: null  # 限制處理圖片數量（null = 全部）

# LayerD Decomposition Step
layerd:
  rtdetr_output_dir: "outputs/pipeline_outputs/rtdetr"  # 讀取 RTDETR 結果
  output_dir: "outputs/pipeline_outputs/layerd"  # LayerD 結果輸出目錄
  max_iterations: 2  # LayerD 分解迭代次數
  device: "cuda"  # "cpu" 或 "cuda"
  limit: null
  matting_process_size: [512, 512]  # 處理尺寸（減少記憶體使用）
  max_image_size: [1536, 1536]  # 最大圖像尺寸（大圖會縮放）

# CLD Format Conversion Step
cld:
  rtdetr_output_dir: "outputs/pipeline_outputs/rtdetr"  # 讀取 RTDETR 結果
  layerd_output_dir: "outputs/pipeline_outputs/layerd"  # 讀取 LayerD 結果
  output_dir: "outputs/pipeline_outputs/cld"  # CLD 格式輸出目錄

# Step 3.5: VLM Caption Generation (可選)
step3_5:
  cld_output_dir: "outputs/pipeline_outputs/cld"  # 讀取 CLD JSON 文件
  force_regenerate: false  # 強制重新生成 captions
  vlm:
    use_vlm_caption: true  # 設為 true 啟用 VLM caption 生成
    vlm_model_id: "liuhaotian/llava-v1.5-7b"  # LLaVA 模型 ID
    vlm_device: "cuda"
    vlm_load_in_4bit: true  # 使用 4-bit 量化節省記憶體
    vlm_max_new_tokens: 96
    vlm_temperature: 0.2
    vlm_prompt: "Describe style, main subject, and especially the background of the whole image in one short sentence."

# Environment names
rtdetr_conda_env: "ultralytics"  # RTDETR 步驟的 conda 環境
cld_conda_env: "CLD"  # CLD 步驟的 conda 環境
vlm_conda_env: "llava"  # VLM 步驟的 conda 環境
```

### CLD Inference 配置 (`configs/exp001/cld/infer.yaml`)

Step 4 需要額外的 CLD inference 配置檔案：

```yaml
seed: 42
max_layer_num: 52
use_pipeline_dataset: true  # 必須設為 true

# 指向 Step 3 的輸出目錄
data_dir: "outputs/pipeline_outputs/cld"

# 模型路徑
pretrained_model_name_or_path: "checkpoints/flux/FLUX.1-dev"
pretrained_adapter_path: "checkpoints/flux/FLUX.1-dev-Controlnet-Inpainting-Alpha"
transp_vae_path: "checkpoints/cld/trans_vae/0008000.pt"
# ... 其他模型路徑

# 輸出目錄
save_dir: "outputs/pipeline_outputs/cld_inference"

# 推理參數
cfg: 4.0
num_inference_steps: 28
```

### 路徑解析規則

- **相對路徑**：相對於配置檔案所在目錄解析為絕對路徑
  - 例如：`configs/exp001/pipeline.yaml` 中的 `"inputs"` 會解析為 `configs/exp001/inputs`
- **絕對路徑**：保持不變
- **好處**：無論在哪個工作目錄執行腳本，都能正確找到檔案

---

## 🎯 執行方式

### 完整 Pipeline（逐步執行）

```bash
# 從專案根目錄執行

# Step 1: RTDETR Detection
python -m src.pipeline.steps.step_rtdetr --config configs/exp001/pipeline.yaml

# Step 2: LayerD Decomposition
python -m src.pipeline.steps.step_layerd --config configs/exp001/pipeline.yaml

# Step 3: CLD Format Conversion
python -m src.pipeline.steps.step_conversion --config configs/exp001/pipeline.yaml

# Step 3.5: VLM Caption Generation (可選)
python -m src.pipeline.steps.step_vlm --config configs/exp001/pipeline.yaml
# 或強制重新生成
python -m src.pipeline.steps.step_vlm --config configs/exp001/pipeline.yaml --force

# Step 4: CLD Inference
python -m src.pipeline.steps.step_cld --config configs/exp001/pipeline.yaml
# 或指定 CLD inference config
python -m src.pipeline.steps.step_cld --config configs/exp001/pipeline.yaml --cld-infer-config configs/exp001/cld/infer.yaml
```

### 單獨執行各步驟

每個步驟都可以獨立執行，只需要確保前置步驟的輸出存在。

#### Step 1: RTDETR Detection

```bash
python -m src.pipeline.steps.step_rtdetr --config configs/exp001/pipeline.yaml
```

**輸出**：`outputs/pipeline_outputs/rtdetr/*.json`（每張圖一個 JSON 檔案）

#### Step 2: LayerD Decomposition

```bash
python -m src.pipeline.steps.step_layerd --config configs/exp001/pipeline.yaml
```

**輸出**：`outputs/pipeline_outputs/layerd/*.npz`（每張圖一個 NPZ 檔案，包含 masks）

#### Step 3: CLD Format Conversion

```bash
python -m src.pipeline.steps.step_conversion --config configs/exp001/pipeline.yaml
```

**輸出**：`outputs/pipeline_outputs/cld/*.json`（每張圖一個 JSON 檔案，包含 CLD 推理所需的格式）

#### Step 3.5: VLM Caption Generation（可選）

```bash
python -m src.pipeline.steps.step_vlm --config configs/exp001/pipeline.yaml
```

**注意**：
- 需要 `llava` conda 環境
- 會讀取 `outputs/pipeline_outputs/cld/*.json` 並更新 `whole_caption` 欄位
- 如果 JSON 中已有 `whole_caption`，會跳過該文件（除非使用 `--force`）

**輸出**：更新後的 `outputs/pipeline_outputs/cld/*.json`（包含 `whole_caption`）

#### Step 4: CLD Inference

```bash
python -m src.pipeline.steps.step_cld --config configs/exp001/pipeline.yaml
```

**注意**：
- 需要 `CLD` conda 環境
- 需要額外的 CLD inference 配置檔案（預設會從 pipeline config 位置推斷）

**輸出**：在 `save_dir` 下生成：
- `case_0/`, `case_1/`, ... - 每個樣本的詳細結果
  - `whole_image_rgba.png` - 完整圖像 RGBA
  - `background_rgba.png` - 背景層 RGBA
  - `layer_0_rgba.png`, `layer_1_rgba.png`, ... - 各前景層 RGBA
  - `origin.png` - 原始輸入圖像
  - `case_0.png` - 最終合成圖像
- `merged/` - 所有合成圖像的 RGB 版本
- `merged_rgba/` - 所有合成圖像的 RGBA 版本

### 環境覆蓋

如果需要覆蓋配置檔案中的環境設定：

```bash
# 使用不同的 conda 環境
python -m src.pipeline.steps.step_rtdetr --config configs/exp001/pipeline.yaml --conda-env my_ultralytics_env
python -m src.pipeline.steps.step_vlm --config configs/exp001/pipeline.yaml --conda-env my_llava_env
```

---

## 📦 輸出格式

### Step 1: RTDETR 輸出 (`*.json`)

```json
{
  "image_path": "path/to/image.png",
  "image_size": [height, width],
  "boxes": [[x1, y1, x2, y2, conf, cls], ...]
}
```

### Step 2: LayerD 輸出 (`*.npz`)

- `masks`: List of numpy arrays (Front-to-Back masks，最後一個是背景)
- `image_size`: [height, width]
- `image_path`: Original image path

### Step 3: CLD 輸出 (`*.json`) - Step 3 後

```json
{
  "image_path": "path/to/image.png",
  "ordered_bboxes": [[x1, y1, x2, y2], ...],
  "quantized_boxes": [[x1, y1, x2, y2], ...],  # 量化到 16 的倍數
  "layer_indices": [1, 2, 3, ...],  # 1-based foreground layers
  "caption": "",  # 空字串
  "whole_caption": "",  # 空字串，等待 Step 3.5 填充
  "debug_info": {...}
}
```

### Step 3.5: CLD 輸出 (`*.json`) - Step 3.5 後

```json
{
  "image_path": "path/to/image.png",
  "ordered_bboxes": [[x1, y1, x2, y2], ...],
  "quantized_boxes": [[x1, y1, x2, y2], ...],
  "layer_indices": [1, 2, 3, ...],
  "caption": "VLM generated caption",
  "whole_caption": "VLM generated caption",  # Step 3.5 生成
  "debug_info": {...}
}
```

### Step 4: CLD Inference 輸出

在 `save_dir` 目錄下：
- `case_0/`, `case_1/`, ... - 每個樣本的詳細結果目錄
- `merged/` - 所有合成圖像的 RGB 版本
- `merged_rgba/` - 所有合成圖像的 RGBA 版本

---

## 🔍 故障排除

### 環境問題

**問題：找不到 conda 環境**
```bash
# 檢查環境是否存在
conda env list

# 如果不存在，重新設置
python scripts/setup_environments.py --cld --ultralytics --llava
```

**問題：uv 未找到**
```bash
# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 驗證安裝
uv --version
```

### 路徑問題

**問題：找不到輸入檔案**
- 檢查配置檔案中的 `input_dir` 路徑
- 確保路徑是相對於配置檔案位置的
- 或使用絕對路徑

**問題：找不到前置步驟的輸出**
- 檢查配置檔案中的 `*_output_dir` 路徑
- 確保前置步驟已成功執行
- 檢查輸出目錄中是否有對應的檔案

### 記憶體問題

**問題：GPU OOM (Out of Memory)**
- 減少 `layerd.max_iterations`（例如從 3 改為 2）
- 減少 `layerd.matting_process_size`（例如從 [1024, 1024] 改為 [512, 512]）
- 減少 `layerd.max_image_size`（例如從 [2048, 2048] 改為 [1536, 1536]）
- 檢查是否有其他進程佔用 GPU 記憶體
- 考慮使用更大的 GPU

**問題：LayerD 記憶體不足**
- 調整 `matting_process_size` 和 `max_image_size` 參數
- 注意：縮小圖像會影響品質，masks 會放大但可能失去細節

### 模型載入問題

**問題：RTDETR 模型載入失敗**
- 檢查 `rtdetr.model_path` 是否正確
- 確保模型檔案存在
- 檢查 conda 環境是否正確安裝 ultralytics

**問題：LLaVA 模型載入失敗**
- 確認 `vlm_model_id` 是否正確
- 檢查 `llava` conda 環境是否正確設置
- 確認 transformers 版本兼容
- 檢查網路連線（需要從 HuggingFace 下載模型）

**問題：CLD 模型載入失敗**
- 檢查 CLD inference config 中的模型路徑
- 確保所有必要的 checkpoint 檔案存在
- 參考 `scripts/download_cld_assets.py` 下載必要的模型

### 執行順序問題

**問題：步驟執行順序錯誤**
- 必須按照順序執行：Step 1 → Step 2 → Step 3 → Step 3.5 (可選) → Step 4
- 每個步驟依賴前一步的輸出

**問題：檔案匹配失敗**
- 各步驟透過檔名（stem）匹配，確保檔名一致
- 例如：`image1.png` → `image1.json` → `image1.npz` → `image1.json`

### 其他問題

**問題：Step 3.5 跳過所有檔案**
- 檢查 `use_vlm_caption` 是否設為 `true`
- 如果 JSON 中已有 `whole_caption`，會自動跳過
- 使用 `--force` 強制重新生成

**問題：CLD inference 找不到 JSON 檔案**
- 檢查 CLD inference config 中的 `data_dir` 是否指向正確的目錄
- 確保 `use_pipeline_dataset: true`
- 檢查 JSON 檔案是否存在且格式正確

---

## 📝 注意事項

1. **路徑處理**：配置檔案中的路徑會自動解析為相對於 config 檔案所在目錄的絕對路徑，這樣無論在哪個工作目錄執行腳本，都能正確找到檔案。

2. **環境管理**：每個步驟使用不同的環境，腳本會自動切換環境，無需手動 `conda activate`。

3. **執行順序**：必須按照順序執行，因為後續步驟依賴前一步的輸出。

4. **檔案匹配**：各步驟透過檔名（stem）匹配，確保檔名一致。

5. **Step 3.5 是可選的**：如果不需要 VLM caption，可以跳過 Step 3.5，CLD 推理會使用空的 caption。

6. **記憶體管理**：LayerD 和 CLD 步驟可能消耗大量 GPU 記憶體，建議根據 GPU 容量調整配置參數。

---

## 🔗 相關文檔

- [環境設置說明](scripts/README_SETUP.md) - 詳細的環境設置指南
- [配置檔案範例](configs/exp001/pipeline.yaml) - Pipeline 配置範例
- [CLD Inference 配置範例](configs/exp001/cld/infer.yaml) - CLD 推理配置範例

---

## 💡 範例工作流程

### 完整流程範例

```bash
# 1. 設置環境（首次使用）
python scripts/setup_environments.py --all

# 2. 準備配置檔案
cp configs/exp001/pipeline.yaml configs/my_exp/pipeline.yaml
# 編輯 configs/my_exp/pipeline.yaml

# 3. 執行完整 pipeline
python -m src.pipeline.steps.step_rtdetr --config configs/my_exp/pipeline.yaml
python -m src.pipeline.steps.step_layerd --config configs/my_exp/pipeline.yaml
python -m src.pipeline.steps.step_conversion --config configs/my_exp/pipeline.yaml
python -m src.pipeline.steps.step_vlm --config configs/my_exp/pipeline.yaml
python -m src.pipeline.steps.step_cld --config configs/my_exp/pipeline.yaml

# 4. 查看結果
ls outputs/pipeline_outputs/cld_inference/
```

### 只執行到 Step 3（不需要 VLM 和 CLD Inference）

```bash
python -m src.pipeline.steps.step_rtdetr --config configs/exp001/pipeline.yaml
python -m src.pipeline.steps.step_layerd --config configs/exp001/pipeline.yaml
python -m src.pipeline.steps.step_conversion --config configs/exp001/pipeline.yaml
```

---

如有其他問題，請參考各步驟的原始碼或提交 issue。

