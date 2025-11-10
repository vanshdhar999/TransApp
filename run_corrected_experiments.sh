#!/bin/bash

#################################################################################################################
#
# CORRECTED FINAL EXPERIMENTS SCRIPT
# Fixes all issues found in the experiment log
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
LOG_FILE="results/corrected_experimental_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Experiment log started at: $(date)" | tee -a "$LOG_FILE"

# Selected appliance cases (consistent across all experiments)
CASES=("cooker_case" "dishwasher_case" "waterheater_case" "laptopcomputer_case" "pluginheater_case")

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🔬 EXPERIMENT 1: Baseline TransApp Model (No Pretraining)${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 Results stored in: results/TransAppResults/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"

cd experiments

for case in "${CASES[@]}"; do
    echo -e "${YELLOW}🎯 Running Baseline TransApp for: ${case}${NC}" | tee -a "../$LOG_FILE"
    
    if python RunTransAppClassif.py "$case" TransApp 96 1.0 >> "../$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed Baseline TransApp for: ${case}${NC}" | tee -a "../$LOG_FILE"
    else
        echo -e "${RED}❌ Error in Baseline TransApp for: ${case}${NC}" | tee -a "../$LOG_FILE"
    fi
    echo "----------------------------------------" | tee -a "../$LOG_FILE"
done

cd ..

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🔬 EXPERIMENT 2: Pre-trained TST Detection${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 Results stored in: results/TransAppResults_TST/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"

cd experiments_tst

for case in "${CASES[@]}"; do
    echo -e "${YELLOW}🎯 Running TST with Pretraining for: ${case}${NC}" | tee -a "../$LOG_FILE"
    
    # Corrected command: case_name model_name dim_model epochs [norm_type]
    if python RunTransAppClassif_TST.py "$case" TransApp_TST_PT 96 15 BatchNorm >> "../$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed TST with Pretraining for: ${case}${NC}" | tee -a "../$LOG_FILE"
    else
        echo -e "${RED}❌ Error in TST with Pretraining for: ${case}${NC}" | tee -a "../$LOG_FILE"
    fi
    echo "----------------------------------------" | tee -a "../$LOG_FILE"
done

cd ..

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🔬 EXPERIMENT 3: TST Without Pretraining (for comparison)${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 Results stored in: results/TransAppResults_TST/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"

cd experiments_tst

for case in "${CASES[@]}"; do
    echo -e "${YELLOW}🎯 Running TST without Pretraining for: ${case}${NC}" | tee -a "../$LOG_FILE"
    
    # Corrected command: case_name model_name dim_model epochs [norm_type]
    if python RunTransAppClassif_TST.py "$case" TransApp_TST 96 15 BatchNorm >> "../$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed TST without Pretraining for: ${case}${NC}" | tee -a "../$LOG_FILE"
    else
        echo -e "${RED}❌ Error in TST without Pretraining for: ${case}${NC}" | tee -a "../$LOG_FILE"
    fi
    echo "----------------------------------------" | tee -a "../$LOG_FILE"
done

cd ..

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🔬 EXPERIMENT 4: PatchTST Detection (Original Weights)${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📁 Results stored in: results/PatchTSTResults/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"

cd experiments_patch

for case in "${CASES[@]}"; do
    echo -e "${YELLOW}🎯 Running PatchTST for: ${case}${NC}" | tee -a "../$LOG_FILE"
    
    # Corrected command: use --cases instead of --case_name --phase
    if python RunPatchTSTClassification.py --cases "$case" --grid quick >> "../$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ Completed PatchTST for: ${case}${NC}" | tee -a "../$LOG_FILE"
    else
        echo -e "${RED}❌ Error in PatchTST for: ${case}${NC}" | tee -a "../$LOG_FILE"
    fi
    echo "----------------------------------------" | tee -a "../$LOG_FILE"
done

cd ..

echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}🎉 ALL EXPERIMENTS COMPLETED!${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}📊 Results Summary:${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  Experiment 1 (Baseline TransApp): results/TransAppResults/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  Experiment 2 (TST with Pretraining): results/TransAppResults_TST/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  Experiment 3 (TST without Pretraining): results/TransAppResults_TST/${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}  Experiment 4 (PatchTST): results/PatchTSTResults/${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo -e "${BLUE}📋 Appliance Cases Tested:${NC}" | tee -a "$LOG_FILE"
for case in "${CASES[@]}"; do
    echo -e "${BLUE}  - ${case}${NC}" | tee -a "$LOG_FILE"
done
echo "" | tee -a "$LOG_FILE"
echo -e "${BLUE}📝 Complete log saved to: $(pwd)/${LOG_FILE}${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}🕒 Experiment completed at: $(date)${NC}" | tee -a "$LOG_FILE"