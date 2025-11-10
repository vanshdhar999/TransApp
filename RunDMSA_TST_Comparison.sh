#!/bin/bash

#################################################################################################################
#
# DMSA-TST Hybrid Comparison Script
# ==================================
#
# This script runs comprehensive experiments comparing:
# 1. Standard TST (no diagonal masking)
# 2. DMSA-TST (with diagonal masking) 
#
# Tests both with and without temporal embeddings across multiple random seeds
#
# Usage: ./RunDMSA_TST_Comparison.sh [case_name] [dim_model] [epochs]
#
#################################################################################################################

# Default parameters
CASE_NAME=${1:-"cooker_case"}
DIM_MODEL=${2:-96}
EPOCHS=${3:-15}
NORM_TYPE="BatchNorm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🧬 DMSA-TST Hybrid Comparison Experiment${NC}"
echo -e "${PURPLE}=======================================${NC}"
echo ""
echo -e "${BLUE}📊 Configuration:${NC}"
echo -e "   Case: ${CASE_NAME}"
echo -e "   Model Dimension: ${DIM_MODEL}"
echo -e "   Epochs: ${EPOCHS}"
echo -e "   Normalization: ${NORM_TYPE}"
echo ""

# Create results directory
RESULTS_DIR="/home/user/vansh/ISP/TransApp/results/TransAppResults_TST"
mkdir -p "$RESULTS_DIR"

# Create logs directory
LOG_DIR="/home/user/vansh/ISP/TransApp/logs/dmsa_tst_experiments"
mkdir -p "$LOG_DIR"

# Generate timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/dmsa_tst_comparison_${CASE_NAME}_${DIM_MODEL}_${TIMESTAMP}.log"

echo -e "${YELLOW}📝 Logging to: ${LOG_FILE}${NC}"
echo ""

# Function to run experiment with logging
run_experiment() {
    local description=$1
    local command=$2
    
    echo -e "${BLUE}🚀 ${description}${NC}"
    echo "Starting: $description" >> "$LOG_FILE"
    echo "Command: $command" >> "$LOG_FILE"
    echo "Timestamp: $(date)" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
    
    # Run the command and capture output
    if eval $command >> "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed: ${description}${NC}"
        echo "SUCCESS: $description" >> "$LOG_FILE"
    else
        echo -e "${RED}❌ Failed: ${description}${NC}"
        echo "FAILED: $description" >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
    echo ""
}

# Change to project directory
cd /home/user/vansh/ISP/TransApp

echo -e "${YELLOW}🔧 Checking Python environment...${NC}"
python3 -c "import torch; import numpy; import sklearn; print('✅ Dependencies available')" || {
    echo -e "${RED}❌ Missing dependencies. Please install torch, numpy, and scikit-learn${NC}"
    exit 1
}

echo -e "${GREEN}✅ Environment check passed${NC}"
echo ""

# Start comprehensive DMSA-TST comparison
echo -e "${PURPLE}🧪 Starting Comprehensive DMSA-TST Comparison${NC}"
echo -e "${PURPLE}=============================================${NC}"

# Run the comprehensive comparison
run_experiment \
    "DMSA-TST Hybrid Comparison Experiment" \
    "python3 experiments_tst/RunDMSA_TST_Comparison.py"

# Check if results were generated
RESULT_FILES=$(find "$RESULTS_DIR" -name "dmsa_tst_comparison_${CASE_NAME}_*_${TIMESTAMP:0:8}*.json" 2>/dev/null)

if [ ! -z "$RESULT_FILES" ]; then
    echo -e "${GREEN}📊 Results Generated:${NC}"
    for file in $RESULT_FILES; do
        echo -e "   📄 $(basename $file)"
        
        # Quick analysis of results
        if command -v jq >/dev/null 2>&1; then
            echo -e "${BLUE}   📈 Quick Results Summary:${NC}"
            
            # Extract best F1-macro scores for each attention type
            jq -r '.all_results[] | "\(.attention_type) (Embed \(.embed_type), Seed \(.random_seed)): F1-macro = \(.results.F1_SCORE_MACRO | tonumber | . * 100 | round / 100)"' "$file" 2>/dev/null | head -8
        fi
        echo ""
    done
else
    echo -e "${YELLOW}⚠️ No result files found with expected pattern${NC}"
    echo -e "${BLUE}📁 Checking results directory: ${RESULTS_DIR}${NC}"
    ls -la "$RESULTS_DIR" | tail -5
fi

echo ""
echo -e "${PURPLE}📊 Experiment Analysis Summary${NC}"
echo -e "${PURPLE}==============================${NC}"

# If jq is available, provide detailed analysis
if command -v jq >/dev/null 2>&1 && [ ! -z "$RESULT_FILES" ]; then
    for file in $RESULT_FILES; do
        echo -e "${BLUE}📄 Analysis of $(basename $file):${NC}"
        
        # Group results by attention type and embedding
        echo -e "${YELLOW}🔍 Performance by Configuration:${NC}"
        
        # Standard TST results
        echo -e "${GREEN}Standard TST:${NC}"
        jq -r '.all_results[] | select(.mask_diag == false) | "  Embed \(.embed_type), Seed \(.random_seed): \(.results.F1_SCORE_MACRO | tonumber | . * 100 | round / 100)"' "$file" 2>/dev/null
        
        # DMSA-TST results  
        echo -e "${GREEN}DMSA-TST:${NC}"
        jq -r '.all_results[] | select(.mask_diag == true) | "  Embed \(.embed_type), Seed \(.random_seed): \(.results.F1_SCORE_MACRO | tonumber | . * 100 | round / 100)"' "$file" 2>/dev/null
        
        echo ""
    done
else
    echo -e "${YELLOW}💡 Install 'jq' for detailed JSON analysis: sudo apt-get install jq${NC}"
fi

echo -e "${GREEN}✅ DMSA-TST Hybrid Comparison Completed!${NC}"
echo ""
echo -e "${BLUE}📁 Results Location: ${RESULTS_DIR}${NC}"
echo -e "${BLUE}📝 Log File: ${LOG_FILE}${NC}"
echo ""
echo -e "${PURPLE}🧬 Next Steps:${NC}"
echo -e "   1. Analyze JSON results to compare DMSA vs Standard attention"
echo -e "   2. Check which configuration (embedding + attention) works best"
echo -e "   3. Consider running additional experiments with different architectures"
echo ""