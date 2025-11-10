#!/bin/bash

# PatchTST Phase 1 Experiment Launcher
# =====================================
# 
# This script launches PatchTST experiments for CER appliance detection
# Includes environment setup, dependency installation, and experiment execution

echo "🚀 PatchTST Phase 1 Experiment Launcher"
echo "========================================"

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_PATCH_DIR="$SCRIPT_DIR"

echo "📍 Experiment directory: $EXPERIMENTS_PATCH_DIR"
echo "📍 Root directory: $ROOT_DIR"

# # Check if virtual environment exists
# if [ ! -d "$ROOT_DIR/venv" ]; then
#     echo "⚠️  Virtual environment not found. Please create one first:"
#     echo "   cd $ROOT_DIR && python -m venv venv"
#     echo "   source venv/bin/activate"
#     echo "   pip install -r requirements.txt"
#     echo "   pip install -r experiments_patch/requirements_patch.txt"
#     exit 1
# fi

# # Activate virtual environment
# echo "🔧 Activating virtual environment..."
# source "$ROOT_DIR/venv/bin/activate"

# # Install additional PatchTST requirements
# echo "📦 Installing PatchTST dependencies..."
# pip install -r "$EXPERIMENTS_PATCH_DIR/requirements_patch.txt" --quiet

# Verify installations
echo "✅ Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "❌ PyTorch installation failed"
    exit 1
}

python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
    echo "❌ Transformers installation failed"
    exit 1
}

python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')" || {
    echo "❌ Scikit-learn installation failed"
    exit 1
}

# Check GPU availability
echo "🖥️ Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "🎮 GPU detected - experiments will run faster!"
else
    echo "🖥️ No GPU detected - experiments will run on CPU"
fi

# Create necessary directories
echo "📁 Creating result directories..."
mkdir -p "$ROOT_DIR/results/PatchTSTResults"
mkdir -p "$ROOT_DIR/logs/patchtst"

# Function to run experiments for a specific case
run_case_experiments() {
    local case_name=$1
    local grid_type=$2
    
    echo ""
    echo "🧪 Starting PatchTST experiments for case: $case_name"
    echo "📊 Grid type: $grid_type"
    echo "⏰ Start time: $(date)"
    
    # Run the experiment
    cd "$EXPERIMENTS_PATCH_DIR"
    if [ "$grid_type" == "quick" ]; then
        echo "1" | python RunPatchTSTClassification.py --case "$case_name"
    else
        echo "2" | python RunPatchTSTClassification.py --case "$case_name"
    fi
    
    echo "✅ Completed experiments for case: $case_name"
    echo "⏰ End time: $(date)"
}

# Main experiment execution
echo ""
echo "🎯 PatchTST Phase 1 Experiment Options:"
echo "1. Quick test (single case, 1 experiment)"
echo "2. Single case focused grid search (81 experiments)"
echo "3. All cases quick test (5 experiments)"
echo "4. All cases focused grid search (405 experiments)"
echo ""

read -p "Select option (1-4): " option

case $option in
    1)
        echo "🔬 Running quick test for cooker case..."
        run_case_experiments "cooker_case" "quick"
        ;;
    2)
        echo "📊 Available cases:"
        echo "- cooker_case"
        echo "- dishwasher_case"
        echo "- tumbledryer_case"
        echo "- tv_greater21inch_case"
        echo "- waterheater_case"
        echo ""
        read -p "Enter case name: " case_name
        echo "🔬 Running full grid search for $case_name..."
        run_case_experiments "$case_name" "full"
        ;;
    3)
        echo "🔬 Running quick tests for all cases..."
        for case in "cooker_case" "dishwasher_case" "tumbledryer_case" "tv_greater21inch_case" "waterheater_case"; do
            run_case_experiments "$case" "quick"
        done
        ;;
    4)
        echo "⚠️  This will run focused experiments for all cases (may take 4-8 hours)!"
        read -p "Are you sure? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo "🔬 Running focused grid search for all cases..."
            for case in "cooker_case" "dishwasher_case" "tumbledryer_case" "tv_greater21inch_case" "waterheater_case"; do
                run_case_experiments "$case" "full"
            done
        else
            echo "❌ Cancelled"
            exit 0
        fi
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "🎉 PatchTST Phase 1 experiments completed!"
echo "📊 Results saved in: $ROOT_DIR/results/PatchTSTResults/"
echo "📋 Logs saved in: $ROOT_DIR/logs/patchtst/"

# Show recent results
echo ""
echo "📈 Recent result files:"
ls -lt "$ROOT_DIR/results/PatchTSTResults/" | head -5

deactivate