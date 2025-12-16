# 環境設置腳本說明

本目錄包含用於設置所有 pipeline 環境的腳本。

## 快速開始

### 設置所有環境

```bash
# 使用 Python 腳本（推薦）
python scripts/setup_environments.py --all

# 或使用 Shell 腳本
bash scripts/setup_environments.sh --all
```

### 設置特定環境

```bash
# 只設置 CLD
python scripts/setup_environments.py --cld

# 設置多個環境
python scripts/setup_environments.py --cld --ultralytics --llava
```

## 環境說明

### 1. CLD 環境
- **類型**: Conda 環境
- **名稱**: `CLD`
- **設置方式**: 從 `third_party/cld/environment.yml` 創建
- **命令**: `conda env create -f environment.yml`

### 2. LayerD 環境
- **類型**: uv 環境
- **設置方式**: 在 `third_party/layerd/` 目錄下執行 `uv sync`
- **命令**: `cd third_party/layerd && uv sync`

### 3. Ultralytics 環境
- **類型**: Conda 環境
- **名稱**: `ultralytics`
- **設置方式**: 
  - `conda create -n ultralytics python -y`
  - `conda run -n ultralytics pip install ultralytics`

### 4. LLaVA 環境
- **類型**: Conda 環境
- **名稱**: `llava`
- **設置方式**:
  - `conda create -n llava python=3.10 -y`
  - `conda run -n llava pip install --upgrade pip`
  - `cd third_party/llava && conda run -n llava pip install -e .`

## 使用選項

### Python 腳本 (`setup_environments.py`)

```bash
python scripts/setup_environments.py [OPTIONS]

選項:
  --all              設置所有環境
  --cld              設置 CLD 環境
  --layerd           設置 LayerD 環境
  --ultralytics      設置 Ultralytics 環境
  --llava            設置 LLaVA 環境
  --force            強制重新創建已存在的環境
  -h, --help         顯示幫助訊息
```

### Shell 腳本 (`setup_environments.sh`)

```bash
bash scripts/setup_environments.sh [OPTIONS]

選項:
  --all              設置所有環境
  --cld              設置 CLD 環境
  --layerd           設置 LayerD 環境
  --ultralytics      設置 Ultralytics 環境
  --llava            設置 LLaVA 環境
  --force            強制重新創建已存在的環境
```

## 範例

### 完整設置流程

```bash
# 1. 確保 submodules 已初始化
git submodule update --init --recursive

# 2. 設置所有環境
python scripts/setup_environments.py --all

# 3. 驗證環境
conda env list  # 應該看到 CLD, ultralytics, llava
```

### 強制重新創建環境

如果環境設置有問題，可以強制重新創建：

```bash
# 重新創建所有環境
python scripts/setup_environments.py --all --force

# 只重新創建 CLD 環境
python scripts/setup_environments.py --cld --force
```

### 逐步設置

```bash
# Step 1: CLD
python scripts/setup_environments.py --cld

# Step 2: Ultralytics
python scripts/setup_environments.py --ultralytics

# Step 3: LayerD
python scripts/setup_environments.py --layerd

# Step 4: LLaVA
python scripts/setup_environments.py --llava
```

## 前置需求

### Conda
- 需要安裝 Anaconda 或 Miniconda
- 驗證: `conda --version`

### uv
- 需要安裝 uv (Python package manager)
- 安裝: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- 驗證: `uv --version`

### Git Submodules
- 確保所有 third_party submodules 已初始化:
  ```bash
  git submodule update --init --recursive
  ```

## 故障排除

### Conda 環境已存在
如果環境已存在，腳本會詢問是否要重新創建。使用 `--force` 可以跳過詢問。

### uv 未找到
確保 uv 已安裝並在 PATH 中：
```bash
# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 添加到 PATH (如果需要的話)
export PATH="$HOME/.cargo/bin:$PATH"
```

### Submodule 未初始化
如果遇到 "directory not found" 錯誤：
```bash
git submodule update --init --recursive
```

### 權限問題
如果遇到權限問題，確保：
- 有寫入 conda 環境目錄的權限
- 有執行腳本的權限（shell 腳本需要 `chmod +x`）

## 驗證設置

設置完成後，可以驗證環境：

```bash
# 檢查 conda 環境
conda env list

# 測試 CLD 環境
conda run -n CLD python --version

# 測試 Ultralytics 環境
conda run -n ultralytics python -c "import ultralytics; print(ultralytics.__version__)"

# 測試 LLaVA 環境
conda run -n llava python --version

# 測試 LayerD (需要 cd 到目錄)
cd third_party/layerd
uv run python --version
```

## 注意事項

1. **環境名稱**: 
   - CLD: `CLD` (大寫)
   - Ultralytics: `ultralytics` (小寫)
   - LLaVA: `llava` (小寫)

2. **LayerD 特殊處理**: 
   - LayerD 使用 uv，不需要 conda 環境
   - 需要在 `third_party/layerd/` 目錄下執行 `uv sync`

3. **LLaVA Python 版本**: 
   - LLaVA 需要 Python 3.10，腳本會自動設置

4. **環境大小**: 
   - 這些環境可能很大（特別是 CLD 和 LLaVA），請確保有足夠的磁碟空間

