#!/usr/bin/env python3

#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia (Modified for TST testing)
# @description : Easy launcher for TST experiments with JSON logging
# @component: experiments_tst/
# @file : launcher_tst.py
#
#################################################################################################################

import subprocess
import sys
import time
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

def create_log_directories():
    """Create directories for experiment logs"""
    log_dir = Path("../results/experiment_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def save_experiment_log(experiment_data, log_dir):
    """Save experiment log to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_type = experiment_data.get('experiment_type', 'unknown')
    filename = f"{experiment_type}_{timestamp}.json"
    log_file = log_dir / filename
    
    with open(log_file, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)
    
    print(f"📄 Experiment log saved to: {log_file}")
    return log_file

def run_command_with_logging(cmd, description, experiment_data, log_dir):
    """Run a command and handle errors with comprehensive logging"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Update experiment data with start time
    experiment_data.update({
        'command': ' '.join(cmd),
        'description': description,
        'start_time': datetime.now().isoformat(),
        'status': 'running'
    })
    
    # Save initial log
    log_file = save_experiment_log(experiment_data, log_dir)
    
    try:
        # Capture output
        start_time = time.time()
        result = subprocess.run(cmd, 
                              check=True, 
                              capture_output=True, 
                              text=True,
                              timeout=3600)  # 1 hour timeout
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update experiment data with success
        experiment_data.update({
            'status': 'completed',
            'end_time': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': True
        })
        
        # Extract metrics from output if available
        metrics = extract_metrics_from_output(result.stdout)
        if metrics:
            experiment_data['extracted_metrics'] = metrics
        
        # Save final log
        save_experiment_log(experiment_data, log_dir)
        
        print(f"✅ {description} completed successfully in {execution_time:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update experiment data with failure
        experiment_data.update({
            'status': 'failed',
            'end_time': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'return_code': e.returncode,
            'stdout': e.stdout if hasattr(e, 'stdout') else '',
            'stderr': e.stderr if hasattr(e, 'stderr') else str(e),
            'success': False,
            'error_message': str(e)
        })
        
        # Save error log
        save_experiment_log(experiment_data, log_dir)
        
        print(f"❌ {description} failed with error code: {e.returncode}")
        return False
        
    except subprocess.TimeoutExpired as e:
        # Handle timeout
        experiment_data.update({
            'status': 'timeout',
            'end_time': datetime.now().isoformat(),
            'execution_time_seconds': 3600,
            'success': False,
            'error_message': 'Process timed out after 1 hour'
        })
        
        save_experiment_log(experiment_data, log_dir)
        print(f"⏰ {description} timed out after 1 hour")
        return False

def extract_metrics_from_output(stdout):
    """Extract performance metrics from command output"""
    metrics = {}
    
    # Look for common patterns in output
    lines = stdout.split('\n')
    for line in lines:
        line = line.strip()
        
        # Extract loss values
        if 'loss' in line.lower() and ':' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    loss_value = float(parts[1].strip().split()[0])
                    metrics['final_loss'] = loss_value
            except (ValueError, IndexError):
                pass
        
        # Extract accuracy values
        if 'accuracy' in line.lower() and ':' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    acc_value = float(parts[1].strip().split()[0])
                    metrics['accuracy'] = acc_value
            except (ValueError, IndexError):
                pass
        
        # Extract F1 score
        if 'f1' in line.lower() and ':' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    f1_value = float(parts[1].strip().split()[0])
                    metrics['f1_score'] = f1_value
            except (ValueError, IndexError):
                pass
        
        # Extract parameter count
        if 'parameters' in line.lower() and any(x in line for x in ['total', 'trainable']):
            try:
                # Extract number from line
                import re
                numbers = re.findall(r'[\d,]+', line)
                if numbers:
                    param_count = int(numbers[0].replace(',', ''))
                    if 'total' in line.lower():
                        metrics['total_parameters'] = param_count
                    elif 'trainable' in line.lower():
                        metrics['trainable_parameters'] = param_count
            except (ValueError, IndexError):
                pass
    
    return metrics

def launch_single_tst_experiment(embed_type, dim_model, norm_type="BatchNorm"):
    """Launch a single TST experiment with logging"""
    log_dir = create_log_directories()
    
    experiment_data = {
        'experiment_type': 'tst_pretraining',
        'configuration': {
            'embed_type': embed_type,
            'dim_model': dim_model,
            'norm_type': norm_type
        },
        'experiment_id': f"tst_{embed_type}_{dim_model}_{norm_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    cmd = [sys.executable, "RunTransAppPretraining_TST.py", str(embed_type), str(dim_model), norm_type]
    description = f"TST experiment: embed_type={embed_type}, dim_model={dim_model}, norm={norm_type}"
    
    return run_command_with_logging(cmd, description, experiment_data, log_dir)

def launch_architecture_comparison(embed_type, dim_model, epochs=10):
    """Launch architecture comparison with logging"""
    log_dir = create_log_directories()
    
    experiment_data = {
        'experiment_type': 'architecture_comparison',
        'configuration': {
            'embed_type': embed_type,
            'dim_model': dim_model,
            'epochs': epochs
        },
        'experiment_id': f"compare_{embed_type}_{dim_model}_{epochs}ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    cmd = [sys.executable, "CompareArchitectures.py", str(embed_type), str(dim_model), str(epochs)]
    description = f"Architecture comparison: embed_type={embed_type}, dim_model={dim_model}"
    
    return run_command_with_logging(cmd, description, experiment_data, log_dir)

def launch_classification_experiment(case_name, model_name, dim_model, epochs, norm_type="BatchNorm"):
    """Launch classification experiment with logging"""
    log_dir = create_log_directories()
    
    experiment_data = {
        'experiment_type': 'tst_classification',
        'configuration': {
            'case_name': case_name,
            'model_name': model_name,
            'dim_model': dim_model,
            'epochs': epochs,
            'norm_type': norm_type
        },
        'experiment_id': f"classif_{case_name}_{model_name}_{dim_model}_{norm_type}_{epochs}ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    cmd = [sys.executable, "RunTransAppClassif_TST.py", case_name, model_name, str(dim_model), str(epochs), norm_type]
    description = f"TST Classification: {case_name} with {model_name} (dim={dim_model}, norm={norm_type})"
    
    return run_command_with_logging(cmd, description, experiment_data, log_dir)

def launch_tst_batch_experiments():
    """Launch batch TST experiments with comprehensive logging"""
    log_dir = create_log_directories()
    
    configurations = [
        (0, 64, "BatchNorm"),
        (0, 64, "LayerNorm"),
        (1, 64, "BatchNorm"),
        (0, 96, "BatchNorm"),
    ]
    
    # Create batch experiment summary
    batch_data = {
        'experiment_type': 'tst_batch',
        'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'start_time': datetime.now().isoformat(),
        'total_experiments': len(configurations),
        'configurations': configurations,
        'individual_experiments': []
    }
    
    successful = 0
    total = len(configurations)
    
    print(f"🎯 Running {total} TST batch experiments...")
    
    for i, (embed_type, dim_model, norm_type) in enumerate(configurations, 1):
        print(f"\n{'='*60}")
        print(f"BATCH EXPERIMENT {i}/{total}")
        print(f"{'='*60}")
        
        # Individual experiment data
        individual_experiment = {
            'batch_index': i,
            'embed_type': embed_type,
            'dim_model': dim_model,
            'norm_type': norm_type,
            'start_time': datetime.now().isoformat()
        }
        
        success = launch_single_tst_experiment(embed_type, dim_model, norm_type)
        
        individual_experiment.update({
            'end_time': datetime.now().isoformat(),
            'success': success
        })
        
        batch_data['individual_experiments'].append(individual_experiment)
        
        if success:
            successful += 1
        
        # Small delay between experiments
        if i < total:
            print("⏱️ Waiting 30 seconds before next experiment...")
            time.sleep(30)
    
    # Finalize batch summary
    batch_data.update({
        'end_time': datetime.now().isoformat(),
        'successful_experiments': successful,
        'failed_experiments': total - successful,
        'success_rate': successful / total
    })
    
    # Save batch summary
    save_experiment_log(batch_data, log_dir)
    
    print(f"\n🎉 Batch experiments completed: {successful}/{total} successful")
    print(f"📊 Success rate: {successful/total*100:.1f}%")

def show_experiment_logs():
    """Show recent experiment logs"""
    log_dir = Path("../results/experiment_logs")
    
    if not log_dir.exists():
        print("📁 No experiment logs found")
        return
    
    log_files = sorted(log_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not log_files:
        print("📁 No experiment logs found")
        return
    
    print(f"\n📋 Recent Experiment Logs ({len(log_files)} total):")
    print("-" * 80)
    
    for i, log_file in enumerate(log_files[:10]):  # Show last 10
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            experiment_type = data.get('experiment_type', 'unknown')
            experiment_id = data.get('experiment_id', log_file.stem)
            status = data.get('status', 'unknown')
            start_time = data.get('start_time', 'unknown')
            
            status_emoji = {
                'completed': '✅',
                'failed': '❌',
                'timeout': '⏰',
                'running': '🔄'
            }.get(status, '❓')
            
            print(f"{i+1:2d}. {status_emoji} {experiment_type} | {experiment_id}")
            print(f"     Started: {start_time} | Status: {status}")
            
            if 'execution_time_seconds' in data:
                exec_time = data['execution_time_seconds']
                print(f"     Duration: {exec_time:.1f}s")
            
            print()
            
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="TST Experiments Launcher with JSON Logging")
    parser.add_argument("mode", choices=["single", "compare", "classify", "batch", "logs"], 
                       help="Experiment mode")
    parser.add_argument("--embed-type", type=int, choices=[0, 1], default=0,
                       help="Embedding type (0=None, 1=Temporal)")
    parser.add_argument("--dim-model", type=int, default=64,
                       help="Model dimension")
    parser.add_argument("--norm-type", choices=["BatchNorm", "LayerNorm"], default="BatchNorm",
                       help="Normalization type for TST")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--case-name", type=str, default="cooker_case",
                       help="Classification case name")
    parser.add_argument("--model-name", type=str, default="TransApp_TST",
                       help="Model name for classification")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("RunTransAppPretraining_TST.py").exists():
        print("❌ Error: Please run this script from the experiments_tst directory")
        print("Current directory should contain RunTransAppPretraining_TST.py")
        sys.exit(1)
    
    print("🏗️ TST Experiments Launcher with JSON Logging")
    print(f"Mode: {args.mode}")
    
    if args.mode == "single":
        print(f"Configuration: embed_type={args.embed_type}, dim_model={args.dim_model}, norm={args.norm_type}")
        launch_single_tst_experiment(args.embed_type, args.dim_model, args.norm_type)
    
    elif args.mode == "compare":
        print(f"Configuration: embed_type={args.embed_type}, dim_model={args.dim_model}, epochs={args.epochs}")
        launch_architecture_comparison(args.embed_type, args.dim_model, args.epochs)
    
    elif args.mode == "classify":
        print(f"Configuration: case={args.case_name}, model={args.model_name}, dim={args.dim_model}")
        launch_classification_experiment(args.case_name, args.model_name, args.dim_model, args.epochs, args.norm_type)
    
    elif args.mode == "batch":
        launch_tst_batch_experiments()
    
    elif args.mode == "logs":
        show_experiment_logs()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        print("🏗️ TST Experiments Launcher - Interactive Mode with JSON Logging")
        print("\n📋 Choose experiment type:")
        print("1. Single TST experiment")
        print("2. Architecture comparison (Standard vs TST)")
        print("3. TST Classification experiment")
        print("4. Batch TST experiments")
        print("5. View experiment logs")
        print("6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                embed_type = int(input("Enter embed_type (0=None, 1=Temporal): "))
                dim_model = int(input("Enter model dimension (e.g., 64, 96): "))
                norm_type = input("Enter norm type (BatchNorm/LayerNorm) [BatchNorm]: ").strip() or "BatchNorm"
                
                launch_single_tst_experiment(embed_type, dim_model, norm_type)
                
            elif choice == "2":
                embed_type = int(input("Enter embed_type (0=None, 1=Temporal): "))
                dim_model = int(input("Enter model dimension (e.g., 64, 96): "))
                epochs = int(input("Enter number of epochs [10]: ") or "10")
                
                launch_architecture_comparison(embed_type, dim_model, epochs)
                
            elif choice == "3":
                case_name = input("Enter case name [cooker_case]: ").strip() or "cooker_case"
                model_name = input("Enter model name [TransApp_TST]: ").strip() or "TransApp_TST"
                dim_model = int(input("Enter model dimension [64]: ") or "64")
                epochs = int(input("Enter number of epochs [15]: ") or "15")
                norm_type = input("Enter norm type (BatchNorm/LayerNorm) [BatchNorm]: ").strip() or "BatchNorm"
                
                launch_classification_experiment(case_name, model_name, dim_model, epochs, norm_type)
                
            elif choice == "4":
                print("\n🚀 Launching batch TST experiments...")
                launch_tst_batch_experiments()
                
            elif choice == "5":
                show_experiment_logs()
                
            elif choice == "6":
                print("👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
            sys.exit(0)
        except ValueError as e:
            print(f"❌ Error: Invalid input. {e}")
            sys.exit(1)
    else:
        main()