#!/bin/bash

#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Vansh Dhar (Based on TransApp framework)
# @description : Complete TST Pre-training and Fine-tuning Pipeline
# @component: root/
# @file : run_complete_tst_pipeline.sh
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
LOG_DIR="logs/tst_pipeline"
PRETRAINED_DIR="results/TransAppPretrained_TST"
FINETUNING_DIR="results/TST_FineTuning"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $PRETRAINED_DIR
mkdir -p $FINETUNING_DIR

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
    
    # Check for CER dataset
    CER_DATA_PATH="/home/user/vansh/ISP/TransApp/data/Inputs/x_residential_25728.csv"
    if [ -f "$CER_DATA_PATH" ]; then
        print_success "CER dataset found"
        return 0
    else
        print_error "CER dataset not found at: $CER_DATA_PATH"
        print_warning "Both pre-training and fine-tuning require the CER dataset"
        print_info "Please ensure you have:"
        print_info "  1. Downloaded the CER dataset"
        print_info "  2. Placed it in the correct directory structure"
        return 1
    fi
}

# Function to run TST pre-training
run_tst_pretraining() {
    local embed_type="$1"
    local dim_model="$2"
    local norm_type="$3"
    
    local exp_id="pretraining_${embed_type}_${dim_model}_${norm_type}_$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/${exp_id}.log"
    
    print_header "TST Pre-training: ${dim_model}D ${norm_type} (Embed: ${embed_type})"
    
    echo -e "${PURPLE}🚀 Pre-training Details:${NC}"
    echo -e "   Embed Type: $embed_type"
    echo -e "   Model Dimension: $dim_model"
    echo -e "   Normalization: $norm_type"
    echo -e "   Log: $log_file"
    echo -e "   Started: $(date)"
    
    # Set PYTHONPATH and run pre-training
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    if $PYTHON_CMD experiments_tst/RunTransAppPretraining_TST.py "$embed_type" "$dim_model" "$norm_type" 2>&1 | tee "$log_file"; then
        print_success "Pre-training completed: ${dim_model}D ${norm_type}"
        return 0
    else
        print_error "Pre-training failed: ${dim_model}D ${norm_type}"
        print_warning "Check log file: $log_file"
        echo -e "\n${RED}Last few lines of error:${NC}"
        tail -n 5 "$log_file" | sed 's/^/  /'
        return 1
    fi
}

# Function to check if pre-trained model exists
pretrained_model_exists() {
    local embed_type="$1"
    local dim_model="$2"
    local norm_type="$3"
    
    local embed_dir=""
    if [ "$embed_type" = "0" ]; then
        embed_dir="None"
    else
        embed_dir="Embed"
    fi
    
    local model_path="$PRETRAINED_DIR/$embed_dir/TransApp_TST_${dim_model}_${norm_type}.pt"
    
    if [ -f "$model_path" ]; then
        return 0  # Model exists
    else
        return 1  # Model doesn't exist
    fi
}

# Function to run comprehensive pre-training
run_comprehensive_pretraining() {
    print_header "Comprehensive TST Pre-training"
    
    if ! check_data_availability; then
        return 1
    fi
    
    # Define pre-training configurations
    local configs=(
        "0 64 BatchNorm"
        "0 96 BatchNorm"
        "0 128 BatchNorm"
        "0 64 LayerNorm"
        "0 96 LayerNorm"
        "1 64 BatchNorm"
        "1 96 BatchNorm"
    )
    
    local total_configs=${#configs[@]}
    local completed=0
    local skipped=0
    local failed=0
    
    print_info "Planned pre-training configurations: $total_configs"
    
    for config in "${configs[@]}"; do
        read -r embed_type dim_model norm_type <<< "$config"
        
        print_info "Processing: Embed=$embed_type, Dim=$dim_model, Norm=$norm_type"
        
        # Check if model already exists
        if pretrained_model_exists "$embed_type" "$dim_model" "$norm_type"; then
            print_warning "Pre-trained model already exists, skipping: ${dim_model}D ${norm_type}"
            ((skipped++))
            continue
        fi
        
        # Run pre-training
        if run_tst_pretraining "$embed_type" "$dim_model" "$norm_type"; then
            ((completed++))
        else
            ((failed++))
        fi
        
        # Small pause between runs
        sleep 2
    done
    
    print_header "Pre-training Summary"
    print_info "📊 Results:"
    print_info "   ✅ Completed: $completed"
    print_info "   ⏭️  Skipped (already exist): $skipped"
    print_info "   ❌ Failed: $failed"
    print_info "   📁 Total configurations: $total_configs"
    
    # List all available pre-trained models
    print_info "🔍 Available pre-trained models:"
    find "$PRETRAINED_DIR" -name "*.pt" -type f | while read model; do
        print_info "   📁 $(basename $model)"
    done
}

# Function to run fine-tuning experiments
run_finetuning_experiments() {
    local experiment_type="$1"  # conservative, moderate, aggressive, or comprehensive
    
    print_header "TST Fine-tuning Experiments: $experiment_type"
    
    # Check if pre-trained models exist
    local model_count=$(find "$PRETRAINED_DIR" -name "*.pt" -type f | wc -l)
    
    if [ $model_count -eq 0 ]; then
        print_error "No pre-trained models found!"
        print_warning "Please run pre-training first:"
        print_info "  $0 pretraining"
        return 1
    fi
    
    print_success "Found $model_count pre-trained models"
    
    # Set up fine-tuning command arguments
    local cmd_args="--pretrained_dir $PRETRAINED_DIR --output_dir $FINETUNING_DIR"
    
    # Configure experiments based on type
    case "$experiment_type" in
        "conservative")
            cmd_args="$cmd_args --cases cooker_case dishwasher_case"
            ;;
        "moderate")
            cmd_args="$cmd_args --cases cooker_case dishwasher_case waterheater_case"
            ;;
        "aggressive"|"comprehensive")
            cmd_args="$cmd_args --cases cooker_case dishwasher_case waterheater_case tumbledryer_case tv_greater21inch_case"
            ;;
    esac
    
    local exp_id="finetuning_${experiment_type}_$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/${exp_id}.log"
    
    print_info "🚀 Starting fine-tuning experiments..."
    print_info "   Type: $experiment_type"
    print_info "   Pre-trained models: $model_count"
    print_info "   Log: $log_file"
    
    # Run fine-tuning
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    if $PYTHON_CMD experiments_tst/RunTSTFineTuning_fixed.py $cmd_args 2>&1 | tee "$log_file"; then
        print_success "Fine-tuning experiments completed: $experiment_type"
        print_info "Results saved in: $FINETUNING_DIR"
        return 0
    else
        print_error "Fine-tuning experiments failed: $experiment_type"
        print_warning "Check log file: $log_file"
        return 1
    fi
}

# Function to analyze results
analyze_results() {
    print_header "Analyzing TST Results"
    
    # Check pre-training results
    local pretrained_count=$(find "$PRETRAINED_DIR" -name "*.pt" -type f | wc -l)
    print_info "📊 Pre-trained Models: $pretrained_count"
    
    if [ $pretrained_count -gt 0 ]; then
        print_info "🔍 Available pre-trained models:"
        find "$PRETRAINED_DIR" -name "*.pt" -type f | while read model; do
            local size=$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model" 2>/dev/null || echo "unknown")
            print_info "   📁 $(basename $model) (${size} bytes)"
        done
    fi
    
    # Check fine-tuning results
    if [ -d "$FINETUNING_DIR" ]; then
        local result_count=$(find "$FINETUNING_DIR" -name "*.json" -type f | wc -l)
        local csv_count=$(find "$FINETUNING_DIR" -name "*.csv" -type f | wc -l)
        
        print_info "📈 Fine-tuning Results:"
        print_info "   JSON files: $result_count"
        print_info "   CSV summaries: $csv_count"
        
        # Show latest comprehensive results
        local latest_comprehensive=$(find "$FINETUNING_DIR" -name "comprehensive_finetuning_results_*.json" -type f | tail -1)
        if [ -n "$latest_comprehensive" ]; then
            print_info "📋 Latest comprehensive results: $(basename $latest_comprehensive)"
            
            # Try to extract key metrics using Python (if available)
            if command -v python3 >/dev/null 2>&1; then
                python3 -c "
import json
try:
    with open('$latest_comprehensive', 'r') as f:
        data = json.load(f)
    total = data.get('total_experiments', 0)
    successful = data.get('successful_experiments', 0)
    print(f'   Experiments: {successful}/{total} successful')
    
    results = data.get('results', [])
    if results:
        f1_scores = [r.get('quantile_metrics', {}).get('f1_macro', 0) for r in results if r.get('quantile_metrics')]
        if f1_scores:
            print(f'   Best F1-Macro: {max(f1_scores):.4f}')
            print(f'   Mean F1-Macro: {sum(f1_scores)/len(f1_scores):.4f}')
except Exception as e:
    print(f'   Error parsing results: {e}')
" 2>/dev/null
            fi
        fi
    else
        print_warning "No fine-tuning results found"
    fi
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}Complete TST Pre-training and Fine-tuning Pipeline${NC}"
    echo -e "${BLUE}=================================================${NC}"
    echo
    echo "Usage: $0 [command] [arguments]"
    echo
    echo "Commands:"
    echo "  pretraining              - Run comprehensive pre-training for all configurations"
    echo "  pretraining-single <embed> <dim> <norm>  - Run single pre-training configuration"
    echo "  finetuning <type>        - Run fine-tuning experiments"
    echo "  full-pipeline            - Run complete pipeline: pre-training → fine-tuning"
    echo "  analyze                  - Analyze current results"
    echo "  check                    - Check prerequisites and current status"
    echo "  clean                    - Clean up old results"
    echo
    echo "Fine-tuning Types:"
    echo "  conservative             - Quick test on 2 appliance cases"
    echo "  moderate                 - Medium test on 3 appliance cases"
    echo "  comprehensive            - Full test on all 5 appliance cases"
    echo
    echo "Examples:"
    echo "  $0 pretraining                          # Create all pre-trained models"
    echo "  $0 pretraining-single 0 96 BatchNorm    # Create specific model"
    echo "  $0 finetuning conservative              # Quick fine-tuning test"
    echo "  $0 finetuning comprehensive             # Full fine-tuning experiments"
    echo "  $0 full-pipeline                        # Complete workflow"
    echo
    echo "Pre-training Configurations:"
    echo "  embed: 0 (no exogenous) or 1 (temporal encoding)"
    echo "  dim: 64, 96, 128 (model dimensions)"
    echo "  norm: BatchNorm, LayerNorm"
}

# Function to run full pipeline
run_full_pipeline() {
    print_header "Complete TST Pipeline: Pre-training → Fine-tuning"
    
    print_info "🔄 Step 1: Comprehensive Pre-training"
    if ! run_comprehensive_pretraining; then
        print_error "Pre-training failed, stopping pipeline"
        return 1
    fi
    
    print_info "🔄 Step 2: Conservative Fine-tuning Test"
    if ! run_finetuning_experiments "conservative"; then
        print_error "Conservative fine-tuning failed, stopping pipeline"
        return 1
    fi
    
    print_info "🔄 Step 3: Comprehensive Fine-tuning"
    if ! run_finetuning_experiments "comprehensive"; then
        print_error "Comprehensive fine-tuning failed"
        return 1
    fi
    
    print_header "Pipeline Completed Successfully!"
    analyze_results
}

# Function to clean up results
clean_results() {
    print_header "Cleaning Up Results"
    
    read -p "🗑️  Remove all pre-trained models and fine-tuning results? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -d "$PRETRAINED_DIR" ]; then
            rm -rf "$PRETRAINED_DIR"
            print_success "Removed pre-trained models directory"
        fi
        
        if [ -d "$FINETUNING_DIR" ]; then
            rm -rf "$FINETUNING_DIR"
            print_success "Removed fine-tuning results directory"
        fi
        
        if [ -d "$LOG_DIR" ]; then
            rm -rf "$LOG_DIR"
            print_success "Removed log directory"
        fi
        
        print_success "Cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Main execution logic
main() {
    case "$1" in
        "pretraining")
            run_comprehensive_pretraining
            ;;
        "pretraining-single")
            if [ $# -ne 4 ]; then
                print_error "Usage: $0 pretraining-single <embed_type> <dim_model> <norm_type>"
                print_info "Example: $0 pretraining-single 0 96 BatchNorm"
                exit 1
            fi
            check_data_availability && run_tst_pretraining "$2" "$3" "$4"
            ;;
        "finetuning")
            if [ -z "$2" ]; then
                print_error "Fine-tuning type required"
                print_info "Usage: $0 finetuning <type>"
                print_info "Types: conservative, moderate, comprehensive"
                exit 1
            fi
            run_finetuning_experiments "$2"
            ;;
        "full-pipeline")
            run_full_pipeline
            ;;
        "analyze")
            analyze_results
            ;;
        "check")
            print_header "Checking Prerequisites and Status"
            check_data_availability
            analyze_results
            ;;
        "clean")
            clean_results
            ;;
        "help"|"-h"|"--help"|"")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"