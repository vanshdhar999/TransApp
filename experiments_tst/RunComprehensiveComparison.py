import os, sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_experiment(case_name, model_name, dim_model, epochs, norm_type="BatchNorm"):
    """Run a single experiment and handle errors"""
    cmd = [
        "python", "experiments_tst/RunTransAppClassif_TST.py",
        case_name, model_name, str(dim_model), str(epochs), norm_type
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        if result.returncode == 0:
            print(f"✅ Completed: {case_name} with {model_name}")
            return True
        else:
            print(f"❌ Failed: {case_name} with {model_name}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {case_name} with {model_name}")
        return False
    except Exception as e:
        print(f"💥 Exception: {case_name} with {model_name} - {e}")
        return False

def main():
    print("🎯 Comprehensive TST-Enhanced TransApp Experiments")
    
    # Experiment configurations
    cases = [
        'cooker_case', 'dishwasher_case', 'waterheater_case', 
        'tumbledryer_case', 'tv_greater21inch_case'
    ]
    
    models = ['TransApp_TST']  # Start with base TST model
    dimensions = [64, 96]
    norm_types = ['BatchNorm', 'LayerNorm']
    epochs = 15
    
    total_experiments = len(cases) * len(models) * len(dimensions) * len(norm_types)
    completed = 0
    failed = 0
    
    print(f"📊 Total experiments to run: {total_experiments}")
    
    start_time = datetime.now()
    
    for case in cases:
        for model in models:
            for dim in dimensions:
                for norm in norm_types:
                    print(f"\n{'='*70}")
                    print(f"Experiment {completed + failed + 1}/{total_experiments}")
                    print(f"Case: {case}, Model: {model}, Dim: {dim}, Norm: {norm}")
                    print(f"{'='*70}")
                    
                    success = run_experiment(case, model, dim, epochs, norm)
                    
                    if success:
                        completed += 1
                    else:
                        failed += 1
                    
                    # Small delay between experiments
                    time.sleep(5)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉 All experiments completed!")
    print(f"✅ Successful: {completed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {duration}")

if __name__ == "__main__":
    main()
