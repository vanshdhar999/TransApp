#!/bin/bash

#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Vansh Dhar (Based on TransApp framework)
# @description : Key TST-Enhanced TransApp Experiments for Mid-Semester Report
# @component: root/
# @file : run_key_experiments.sh
#
#################################################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_CMD="python"
LOG_DIR="logs"

# Create necessary directories
mkdir -p $LOG_DIR

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to check data availability
check_data_availability() {
    print_info "Checking data availability..."
    
    # Check for CER dataset - Fixed path
    CER_DATA_PATH="/home/user/vansh/ISP/TransApp/data/Inputs/x_residential_25728.csv"
    if [ -f "$CER_DATA_PATH" ]; then
        print_success "CER dataset found"
        return 0
    else
        print_error "CER dataset not found at: $CER_DATA_PATH"
        print_warning "The experiments require the CER dataset to be properly set up"
        print_info "Please ensure you have:"
        print_info "  1. Downloaded the CER dataset"
        print_info "  2. Placed it in the correct directory structure: TransApp/data/Inputs/"
        print_info "  3. Updated the data paths in data_utils.py if needed"
        return 1
    fi
}

# Function to run command with logging and better error handling (FIXED)
run_experiment() {
    local case_name="$1"
    local model_name="$2"
    local dim_model="$3"
    local epochs="$4"
    local norm_type="$5"
    
    local exp_id="${case_name}_${model_name}_${dim_model}_${norm_type}_${epochs}ep"
    local log_file="$LOG_DIR/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${PURPLE}🚀 Running: $exp_id${NC}"
    
    # Check if data is available before running
    if ! check_data_availability > /dev/null 2>&1; then
        print_error "Data not available - skipping $exp_id"
        echo "SKIPPED: Data not available" > "$log_file"
        return 2  # Return 2 for skipped due to data issues
    fi
    
    # Run command with real-time output AND logging
    echo -e "${CYAN}Starting experiment at $(date)${NC}"
    
    # Use tee to show output AND save to log
    if (cd "$(pwd)" && PYTHONPATH="$(pwd):$PYTHONPATH" $PYTHON_CMD experiments_tst/RunTransAppClassif_TST.py "$case_name" "$model_name" "$dim_model" "$epochs" "$norm_type" 2>&1 | tee "$log_file"); then
        print_success "$exp_id completed at $(date)"
        return 0
    else
        local exit_code=${PIPESTATUS[0]}  # Get exit code from python command, not tee
        print_error "$exp_id failed at $(date) - check $log_file"
        # Show last few lines of error for debugging
        echo -e "\n${RED}Last few lines of error:${NC}"
        tail -n 5 "$log_file" | sed 's/^/  /'
        return 1
    fi
}

# Function to check if results exist
results_exist() {
    local case_name="$1"
    local model_name="$2"
    local dim_model="$3"
    local norm_type="$4"
    
    # Check if result files exist
    if find results/TransAppResults_TST -name "*${case_name}*${model_name}*${dim_model}*${norm_type}*results.json" 2>/dev/null | grep -q .; then
        return 0  # Results exist
    else
        return 1  # No results
    fi
}

# Function to setup data environment (mock data for testing)
setup_mock_data() {
    print_header "SETTING UP MOCK DATA FOR TESTING"
    
    print_warning "CER dataset not found - setting up mock data for testing..."
    
    # Create mock data directory structure - Fixed path
    mkdir -p /home/user/vansh/ISP/TransApp/data/Inputs
    
    # Create a simple mock script to generate test data
    cat > /home/user/vansh/ISP/TransApp/data/setup_mock_data.py << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

def create_mock_cer_data():
    """Create mock CER dataset for testing"""
    np.random.seed(42)
    
    # Create mock consumption data (100 customers, 25728 time points)
    n_customers = 100
    n_timepoints = 25728
    
    # Generate realistic consumption patterns
    base_consumption = np.random.exponential(2.0, (n_customers, n_timepoints))
    
    # Add some daily/weekly patterns
    time_idx = np.arange(n_timepoints)
    daily_pattern = 0.5 * np.sin(2 * np.pi * time_idx / 48)  # 48 points per day (30min intervals)
    weekly_pattern = 0.3 * np.sin(2 * np.pi * time_idx / (48 * 7))  # Weekly pattern
    
    for i in range(n_customers):
        base_consumption[i] += daily_pattern + weekly_pattern
        base_consumption[i] = np.maximum(base_consumption[i], 0.1)  # Minimum consumption
    
    # Create DataFrame
    customer_ids = [f'customer_{i:04d}' for i in range(n_customers)]
    time_cols = [f't_{i:05d}' for i in range(n_timepoints)]
    
    df = pd.DataFrame(base_consumption, index=customer_ids, columns=time_cols)
    df.index.name = 'id_pdl'
    
    return df

def create_mock_labels():
    """Create mock appliance labels"""
    np.random.seed(42)
    
    appliances = [
        'cooker_case', 'dishwasher_case', 'waterheater_case', 
        'tumbledryer_case', 'tv_greater21inch_case'
    ]
    
    customer_ids = [f'customer_{i:04d}' for i in range(100)]
    
    labels_data = {}
    for appliance in appliances:
        # Random binary labels with different prevalence
        if 'cooker' in appliance:
            prob = 0.8  # High prevalence
        elif 'tv' in appliance:
            prob = 0.7
        else:
            prob = 0.6
            
        labels = np.random.binomial(1, prob, len(customer_ids))
        labels_data[appliance] = labels
    
    df_labels = pd.DataFrame(labels_data, index=customer_ids)
    df_labels.index.name = 'id_pdl'
    
    return df_labels

if __name__ == "__main__":
    print("Creating mock CER dataset...")
    
    # Create consumption data - Fixed paths
    consumption_df = create_mock_cer_data()
    consumption_df.to_csv('/home/user/vansh/ISP/TransApp/data/Inputs/x_residential_25728.csv')
    print(f"✅ Created mock consumption data: {consumption_df.shape}")
    
    # Create labels
    labels_df = create_mock_labels()
    labels_df.to_csv('/home/user/vansh/ISP/TransApp/data/Inputs/y_residential.csv')
    print(f"✅ Created mock labels data: {labels_df.shape}")
    
    print("🎉 Mock CER dataset created successfully!")
    print("⚠️  Note: This is synthetic data for testing purposes only")
EOF

    # Run the mock data creation
    if python /home/user/vansh/ISP/TransApp/data/setup_mock_data.py; then
        print_success "Mock data created successfully"
        print_warning "Using synthetic data - results will be for testing only"
        return 0
    else
        print_error "Failed to create mock data"
        return 1
    fi
}

# Function to run baseline TransApp experiments (for comparison) - FIXED
run_baseline_experiments() {
    print_header "BASELINE: STANDARD TRANSAPP EXPERIMENTS (~60 minutes)"
    
    print_info "Running standard TransApp experiments for comparison with TST"
    print_info "These provide the baseline performance metrics"
    
    # Check data availability first
    if ! check_data_availability; then
        print_warning "CER dataset not available - cannot run baseline experiments"
        return 1
    fi
    
    # Baseline cases (same as TST experiments for fair comparison)
    local cases=("cooker_case" "dishwasher_case" "waterheater_case")
    local model="TransApp"  # Standard TransApp model
    local dim_model=64
    local epochs=5
    
    local total_exp=${#cases[@]}
    local current_exp=0
    local failed_exp=0
    
    for case in "${cases[@]}"; do
        current_exp=$((current_exp + 1))
        echo -e "\n${YELLOW}📊 Baseline Experiment $current_exp/$total_exp${NC}"
        
        # Check if baseline results already exist
        if find results/TransAppResults -name "*${case}*${model}*${dim_model}*" 2>/dev/null | grep -q .; then
            print_info "Skipping baseline $case (results exist)"
            continue
        fi
        
        echo -e "${PURPLE}🚀 Running baseline: ${case}_${model}_${dim_model}_${epochs}ep${NC}"
        echo -e "${CYAN}Starting baseline experiment at $(date)${NC}"
        
        local log_file="$LOG_DIR/baseline_${case}_${model}_${dim_model}_$(date +%Y%m%d_%H%M%S).log"
        
        # Run command with real-time output AND logging
        if (cd "$(pwd)" && PYTHONPATH="$(pwd):$PYTHONPATH" $PYTHON_CMD experiments/RunTransAppClassif.py "$case" "$model" "$dim_model" "full" 2>&1 | tee "$log_file"); then
            print_success "Baseline completed: $case at $(date)"
        else
            failed_exp=$((failed_exp + 1))
            print_error "Baseline failed: $case at $(date) - check $log_file"
            # Show last few lines of error
            echo -e "\n${RED}Last few lines of error:${NC}"
            tail -n 5 "$log_file" | sed 's/^/  /'
        fi
        
        sleep 3
    done
    
    echo -e "\n${BLUE}📋 BASELINE SUMMARY${NC}"
    echo "Completed: $((current_exp - failed_exp)) baseline experiments"
    echo "Failed: $failed_exp baseline experiments"
}

# Enhanced Phase 1: Core Comparison with Baselines (FIXED)
run_core_experiments() {
    print_header "PHASE 1: CORE TST VS BASELINE COMPARISON (~90 minutes)"
    
    # Check data availability first
    if ! check_data_availability; then
        print_warning "CER dataset not available"
        read -p "Would you like to create mock data for testing? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if ! setup_mock_data; then
                print_error "Cannot proceed without data"
                return 1
            fi
        else
            print_error "Cannot run experiments without data"
            return 1
        fi
    fi
    
    print_info "Running comprehensive TST vs Standard TransApp comparison"
    print_info "Cases: cooker_case, dishwasher_case, waterheater_case"
    print_info "This will provide direct performance comparisons for the report"
    
    # Show experiment plan
    echo -e "\n${CYAN}📋 EXPERIMENT PLAN:${NC}"
    echo "  Phase 1a: Baseline TransApp experiments (3 cases)"
    echo "  Phase 1b: TST experiments (3 cases × 2 dims × 2 norms = 12 experiments)"
    echo "  Total Phase 1: ~15 experiments"
    echo
    
    # First run baseline experiments if needed
    print_info "Step 1: Ensuring baseline TransApp results exist..."
    run_baseline_experiments
    
    print_info "Step 2: Running TST experiments..."
    
    # TST experiments with multiple configurations
    local cases=("cooker_case" "dishwasher_case" "waterheater_case")
    local model="TransApp_TST"
    local dim_models=(32 64)  # Test different model sizes
    local epochs=5
    local norm_types=("BatchNorm" "LayerNorm")
    
    local total_exp=$((${#cases[@]} * ${#dim_models[@]} * ${#norm_types[@]}))
    local current_exp=0
    local failed_exp=0
    local skipped_exp=0
    
    echo -e "\n${CYAN}🎯 Starting TST experiments: $total_exp total${NC}"
    
    for case in "${cases[@]}"; do
        for dim_model in "${dim_models[@]}"; do
            for norm in "${norm_types[@]}"; do
                current_exp=$((current_exp + 1))
                echo -e "\n${YELLOW}📊 TST Experiment $current_exp/$total_exp${NC}"
                echo -e "${YELLOW}   Configuration: $case, $model, dim=$dim_model, norm=$norm${NC}"
                
                if results_exist "$case" "$model" "$dim_model" "$norm"; then
                    print_info "Skipping $case $model $dim_model $norm (results exist)"
                    continue
                fi
                
                # Show progress
                local percent=$((current_exp * 100 / total_exp))
                echo -e "${CYAN}Progress: [$current_exp/$total_exp] ${percent}%${NC}"
                
                case $(run_experiment "$case" "$model" "$dim_model" "$epochs" "$norm") in
                    0) print_success "Completed: $case with $model $dim_model $norm" ;;
                    1) failed_exp=$((failed_exp + 1))
                       print_error "Failed: $case with $model $dim_model $norm" ;;
                    2) skipped_exp=$((skipped_exp + 1))
                       print_warning "Skipped: $case with $model $dim_model $norm (data issues)" ;;
                esac
                
                # Show remaining time estimate
                if [ $current_exp -gt 0 ]; then
                    local remaining=$((total_exp - current_exp))
                    echo -e "${CYAN}Remaining experiments: $remaining${NC}"
                fi
                
                sleep 2
            done
        done
    done
    
    echo -e "\n${BLUE}📋 PHASE 1 SUMMARY${NC}"
    echo "Completed TST: $((current_exp - failed_exp - skipped_exp)) experiments"
    echo "Failed TST: $failed_exp experiments"
    echo "Skipped TST: $skipped_exp experiments"
    echo "Baseline experiments: Available for comparison"
    echo "Phase 1 completed at: $(date)"
}

# Comprehensive experiment runner (FIXED)
run_comprehensive_experiments() {
    print_header "COMPREHENSIVE EXPERIMENTS FOR MID-SEMESTER REPORT"
    print_info "This will run all necessary experiments for a complete analysis"
    print_info "Estimated time: ~4-5 hours"
    
    read -p "Continue with comprehensive experiments? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "🎯 Starting comprehensive experimental suite..."
        
        # Run all phases
        run_core_experiments        # TST vs Baseline comparison
        sleep 10
        run_extended_experiments    # Additional appliance cases
        sleep 10  
        run_ablation_experiments    # TST configuration analysis
        sleep 10
        run_pretraining_experiments # Pretraining benefits
        sleep 10
        run_pretrained_evaluation   # Pretrained model evaluation
        
        print_success "🎉 All comprehensive experiments completed!"
        print_info "📊 You now have:"
        print_info "   ✅ Direct TST vs TransApp comparisons"
        print_info "   ✅ Multiple appliance case studies"
        print_info "   ✅ TST configuration ablations"
        print_info "   ✅ Pretraining benefit analysis"
        print_info "   ✅ Statistical significance data"
        print_info ""
        print_info "📔 Run your MidSemesterReport.ipynb to generate analysis and plots"
    else
        print_info "Comprehensive experiments cancelled"
    fi
}

# Updated main execution logic (FIXED)
case "${1:-help}" in
    "baseline")
        run_baseline_experiments
        ;;
    "core")
        echo -e "${GREEN}🎯 Starting Core TST vs Baseline Experiments${NC}"
        echo -e "${GREEN}Estimated time: ~90 minutes${NC}"
        echo -e "${GREEN}You will see real-time output from each experiment${NC}"
        echo
        run_core_experiments
        ;;
    "extended")
        run_extended_experiments
        ;;
    "ablation")
        run_ablation_experiments
        ;;
    "pretrain")
        run_pretraining_experiments
        ;;
    "pretrained")
        run_pretrained_evaluation
        ;;
    "comprehensive")
        run_comprehensive_experiments
        ;;
    "progress")
        show_experiment_progress
        ;;
    "setup-test")
        test_data_setup
        ;;
    "status")
        show_status
        ;;
    "help"|*)
        show_menu
        ;;
esac

echo -e "\n${GREEN}🎉 Script execution completed!${NC}"
echo -e "📋 Check logs/ directory for detailed experiment logs"
echo -e "📊 Check results/ directory for experimental data"
echo -e "📔 Run your MidSemesterReport.ipynb notebook to generate analysis and plots"