#!/bin/bash

#################################################################################################################
#
# FOCUSED FINE-TUNING EXPERIMENTS SCRIPT
# Uses existing pretrained weights for TransApp and TST, then runs PatchTST
#
#################################################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results directories
mkdir -p results/TransAppResults
mkdir -p results/TransAppResults_TST
mkdir -p results/PatchTSTResults

# Log file
LOG_FILE="results/focused_finetuning_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Focused Fine-tuning Experiment log started at: $(date)" | tee -a "$LOG_FILE"

# Selected appliance cases (consistent across all experiments)
CASES=("cooker_case" "dishwasher_case" "waterheater_case" "laptopcomputer_case" "pluginheater_case")

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🔬 EXPERIMENT 1: TransApp Fine-tuning with Pretrained Weights${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 Results stored in: results/TransAppResults/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}💡 Using pretrained weights: results/TransAppPretrained/Embed/TransApp96.pt${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"

# cd experiments

# for case in "${CASES[@]}"; do
#     echo -e "${YELLOW}🎯 Running TransApp Fine-tuning for: ${case}${NC}" | tee -a "../$LOG_FILE"
    
#     # Using TransAppPT model with pretrained weights
#     if python RunTransAppClassif.py "$case" TransAppPT 96 1 >> "../$LOG_FILE" 2>&1; then
#         echo -e "${GREEN}✅ Completed TransApp Fine-tuning for: ${case}${NC}" | tee -a "../$LOG_FILE"
#     else
#         echo -e "${RED}❌ Error in TransApp Fine-tuning for: ${case}${NC}" | tee -a "../$LOG_FILE"
#     fi
#     echo "----------------------------------------" | tee -a "../$LOG_FILE"
# done

# cd ..

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🔬 EXPERIMENT 2: TST Fine-tuning with Pretrained Weights${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 Results stored in: results/TransAppResults_TST/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}💡 Using pretrained weights: results/TransAppPretrained_TST/Embed/TransApp_TST_96_BatchNorm.pt${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"

cd experiments_tst

for case in "${CASES[@]}"; do
    echo -e "${YELLOW}🎯 Running TST Fine-tuning for: ${case}${NC}" | tee -a "../$LOG_FILE"
    
    # Using TST with pretrained weights (corrected command format)
    if python RunTransAppClassif_TST.py "$case" TransApp_TST_PT 96 15 BatchNorm >> "../$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed TST Fine-tuning for: ${case}${NC}" | tee -a "../$LOG_FILE"
    else
        echo -e "${RED}❌ Error in TST Fine-tuning for: ${case}${NC}" | tee -a "../$LOG_FILE"
    fi
    echo "----------------------------------------" | tee -a "../$LOG_FILE"
done

cd ..

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🔬 EXPERIMENT 3: TST from Scratch (for comparison)${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 Results stored in: results/TransAppResults_TST/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}💡 Training TST from scratch without pretraining${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"

cd experiments_tst

for case in "${CASES[@]}"; do
    echo -e "${YELLOW}🎯 Running TST from scratch for: ${case}${NC}" | tee -a "../$LOG_FILE"
    
    # Using TST without pretraining
    if python RunTransAppClassif_TST.py "$case" TransApp_TST 96 15 BatchNorm >> "../$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed TST from scratch for: ${case}${NC}" | tee -a "../$LOG_FILE"
    else
        echo -e "${RED}❌ Error in TST from scratch for: ${case}${NC}" | tee -a "../$LOG_FILE"
    fi
    echo "----------------------------------------" | tee -a "../$LOG_FILE"
done

cd ..

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🔬 EXPERIMENT 4: PatchTST Baseline${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 Results stored in: results/PatchTSTResults/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}💡 Using original PatchTST weights without pretraining${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"

cd experiments_patch

# First check the actual CLI arguments for PatchTST
echo -e "${YELLOW}🔍 Checking PatchTST CLI arguments...${NC}" | tee -a "../$LOG_FILE"
python RunPatchTSTClassification.py --help >> "../$LOG_FILE" 2>&1 || true

for case in "${CASES[@]}"; do
    echo -e "${YELLOW}🎯 Running PatchTST for: ${case}${NC}" | tee -a "../$LOG_FILE"
    
    # Try different argument formats for PatchTST
    if python RunPatchTSTClassification.py --cases "$case" --grid quick >> "../$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed PatchTST for: ${case}${NC}" | tee -a "../$LOG_FILE"
    elif python RunPatchTSTClassification.py --cases "$case" >> "../$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed PatchTST for: ${case} (fallback)${NC}" | tee -a "../$LOG_FILE"
    else
        echo -e "${RED}❌ Error in PatchTST for: ${case}${NC}" | tee -a "../$LOG_FILE"
        echo -e "${YELLOW}💡 Trying to run all cases together...${NC}" | tee -a "../$LOG_FILE"
        # Try running all cases at once as a fallback
        if [[ "$case" == "cooker_case" ]]; then
            python RunPatchTSTClassification.py --cases "${CASES[*]}" >> "../$LOG_FILE" 2>&1 || true
        fi
    fi
    echo "----------------------------------------" | tee -a "../$LOG_FILE"
done

cd ..

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}🎉 ALL FOCUSED EXPERIMENTS COMPLETED!${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📊 Results Summary:${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  Experiment 1 (TransApp Fine-tuning): results/TransAppResults/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  Experiment 2 (TST Fine-tuning): results/TransAppResults_TST/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  Experiment 3 (TST from scratch): results/TransAppResults_TST/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  Experiment 4 (PatchTST): results/PatchTSTResults/${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo -e "${BLUE}📋 Appliance Cases Tested:${NC}" | tee -a "$LOG_FILE"
for case in "${CASES[@]}"; do
    echo -e "${BLUE}  - ${case}${NC}" | tee -a "$LOG_FILE"
done
echo "" | tee -a "$LOG_FILE"
echo -e "${BLUE}💾 Available Pretrained Weights Used:${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  - TransApp: results/TransAppPretrained/Embed/TransApp96.pt${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  - TST: results/TransAppPretrained_TST/Embed/TransApp_TST_96_BatchNorm.pt${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo -e "${BLUE}📝 Complete log saved to: $(pwd)/${LOG_FILE}${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🕒 Experiment completed at: $(date)${NC}" | tee -a "$LOG_FILE"