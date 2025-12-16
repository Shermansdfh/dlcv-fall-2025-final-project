#!/bin/bash
# Setup script for all pipeline environments (shell version)
#
# Usage:
#   bash scripts/setup_environments.sh [--all|--cld|--layerd|--ultralytics|--llava] [--force]

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SETUP_ALL=false
SETUP_CLD=false
SETUP_LAYERD=false
SETUP_ULTRALYTICS=false
SETUP_LLAVA=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            SETUP_ALL=true
            shift
            ;;
        --cld)
            SETUP_CLD=true
            shift
            ;;
        --layerd)
            SETUP_LAYERD=true
            shift
            ;;
        --ultralytics)
            SETUP_ULTRALYTICS=true
            shift
            ;;
        --llava)
            SETUP_LLAVA=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If --all is specified, set all flags
if [ "$SETUP_ALL" = true ]; then
    SETUP_CLD=true
    SETUP_LAYERD=true
    SETUP_ULTRALYTICS=true
    SETUP_LLAVA=true
fi

# If no environment is selected, show usage
if [ "$SETUP_CLD" = false ] && [ "$SETUP_LAYERD" = false ] && [ "$SETUP_ULTRALYTICS" = false ] && [ "$SETUP_LLAVA" = false ]; then
    echo "Usage: $0 [--all|--cld|--layerd|--ultralytics|--llava] [--force]"
    exit 1
fi

echo "🚀 Starting environment setup..."
echo "   Repository root: $REPO_ROOT"
echo ""

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}❌ Conda not found. Please install conda first.${NC}"
        return 1
    fi
    return 0
}

# Function to check if uv is available
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uv not found. Please install uv first.${NC}"
        echo "   Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi
    return 0
}

# Setup CLD
if [ "$SETUP_CLD" = true ]; then
    echo "============================================================"
    echo "Setting up CLD conda environment"
    echo "============================================================"
    
    if ! check_conda; then
        exit 1
    fi
    
    ENV_YML="$REPO_ROOT/third_party/cld/environment.yml"
    if [ ! -f "$ENV_YML" ]; then
        echo -e "${RED}❌ CLD environment.yml not found: $ENV_YML${NC}"
        exit 1
    fi
    
    # Check if environment exists
    if conda env list | grep -q "^CLD "; then
        if [ "$FORCE" = true ]; then
            echo "🗑️  Removing existing CLD environment..."
            conda env remove -n CLD -y || true
        else
            echo -e "${YELLOW}ℹ️  CLD conda environment already exists. Use --force to recreate.${NC}"
            read -p "   Do you want to recreate it? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "   Skipping CLD environment setup."
            else
                echo "🗑️  Removing existing CLD environment..."
                conda env remove -n CLD -y || true
            fi
        fi
    fi
    
    if ! conda env list | grep -q "^CLD "; then
        echo "📦 Creating CLD conda environment from $ENV_YML..."
        conda env create -f "$ENV_YML"
        echo -e "${GREEN}✅ CLD environment created successfully${NC}"
    fi
fi

# Setup LayerD
if [ "$SETUP_LAYERD" = true ]; then
    echo ""
    echo "============================================================"
    echo "Setting up LayerD uv environment"
    echo "============================================================"
    
    if ! check_uv; then
        exit 1
    fi
    
    LAYERD_DIR="$REPO_ROOT/third_party/layerd"
    if [ ! -d "$LAYERD_DIR" ]; then
        echo -e "${RED}❌ LayerD directory not found: $LAYERD_DIR${NC}"
        echo "   Please ensure third_party/layerd submodule is initialized"
        exit 1
    fi
    
    if [ ! -f "$LAYERD_DIR/pyproject.toml" ]; then
        echo -e "${RED}❌ LayerD pyproject.toml not found${NC}"
        exit 1
    fi
    
    echo "📦 Syncing LayerD uv environment in $LAYERD_DIR..."
    cd "$LAYERD_DIR"
    uv sync
    cd "$REPO_ROOT"
    echo -e "${GREEN}✅ LayerD uv environment synced successfully${NC}"
fi

# Setup Ultralytics
if [ "$SETUP_ULTRALYTICS" = true ]; then
    echo ""
    echo "============================================================"
    echo "Setting up Ultralytics conda environment"
    echo "============================================================"
    
    if ! check_conda; then
        exit 1
    fi
    
    # Check if environment exists
    if conda env list | grep -q "^ultralytics "; then
        if [ "$FORCE" = true ]; then
            echo "🗑️  Removing existing ultralytics environment..."
            conda env remove -n ultralytics -y || true
        else
            echo -e "${YELLOW}ℹ️  Ultralytics conda environment already exists. Use --force to recreate.${NC}"
            read -p "   Do you want to recreate it? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "   Skipping Ultralytics environment setup."
            else
                echo "🗑️  Removing existing ultralytics environment..."
                conda env remove -n ultralytics -y || true
            fi
        fi
    fi
    
    if ! conda env list | grep -q "^ultralytics "; then
        echo "📦 Creating ultralytics conda environment..."
        conda create -n ultralytics python -y
        
        echo "📦 Installing ultralytics package..."
        conda run -n ultralytics pip install ultralytics
        
        echo -e "${GREEN}✅ Ultralytics environment created successfully${NC}"
    fi
fi

# Setup LLaVA
if [ "$SETUP_LLAVA" = true ]; then
    echo ""
    echo "============================================================"
    echo "Setting up LLaVA conda environment"
    echo "============================================================"
    
    if ! check_conda; then
        exit 1
    fi
    
    LLAVA_DIR="$REPO_ROOT/third_party/llava"
    if [ ! -d "$LLAVA_DIR" ]; then
        echo -e "${RED}❌ LLaVA directory not found: $LLAVA_DIR${NC}"
        echo "   Please ensure third_party/llava submodule is initialized"
        exit 1
    fi
    
    # Check for either pyproject.toml or setup.py (both work with pip install -e .)
    if [ ! -f "$LLAVA_DIR/pyproject.toml" ] && [ ! -f "$LLAVA_DIR/setup.py" ]; then
        echo -e "${RED}❌ LLaVA package file not found: neither pyproject.toml nor setup.py exists${NC}"
        exit 1
    fi
    
    # Check if environment exists
    if conda env list | grep -q "^llava "; then
        if [ "$FORCE" = true ]; then
            echo "🗑️  Removing existing llava environment..."
            conda env remove -n llava -y || true
        else
            echo -e "${YELLOW}ℹ️  LLaVA conda environment already exists. Use --force to recreate.${NC}"
            read -p "   Do you want to recreate it? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "   Skipping LLaVA environment setup."
            else
                echo "🗑️  Removing existing llava environment..."
                conda env remove -n llava -y || true
            fi
        fi
    fi
    
    if ! conda env list | grep -q "^llava "; then
        echo "📦 Creating llava conda environment with Python 3.10..."
        conda create -n llava python=3.10 -y
        
        echo "📦 Upgrading pip..."
        conda run -n llava pip install --upgrade pip
        
        echo "📦 Installing LLaVA from $LLAVA_DIR..."
        cd "$LLAVA_DIR"
        conda run -n llava pip install -e .
        cd "$REPO_ROOT"
        
        echo -e "${GREEN}✅ LLaVA environment created successfully${NC}"
    fi
fi

echo ""
echo "============================================================"
echo "Setup Summary"
echo "============================================================"
echo -e "${GREEN}🎉 All selected environments set up successfully!${NC}"

