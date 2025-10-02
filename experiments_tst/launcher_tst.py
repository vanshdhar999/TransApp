#!/usr/bin/env python3

#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia (Modified for TST testing)
# @description : Easy launcher for TST experiments
# @component: experiments_tst/
# @file : launcher_tst.py
#
#################################################################################################################

import subprocess
import sys
import time
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code: {e.returncode}")
        return False

def launch_single_tst_experiment(embed_type, dim_model, norm_type="BatchNorm"):
    """Launch a single TST experiment"""
    cmd = [sys.executable, "RunTransAppPretraining_TST.py", str(embed_type), str(dim_model), norm_type]
    return run_command(cmd, f"TST experiment: embed_type={embed_type}, dim_model={dim_model}, norm={norm_type}")

def launch_architecture_comparison(embed_type, dim_model, epochs=10):
    """Launch architecture comparison"""
    cmd = [sys.executable, "CompareArchitectures.py", str(embed_type), str(dim_model), str(epochs)]
    return run_command(cmd, f"Architecture comparison: embed_type={embed_type}, dim_model={dim_model}")

def launch_tst_batch_experiments():
    """Launch batch TST experiments"""
    
    configurations = [
        (0, 64, "BatchNorm"),
        (0, 64, "LayerNorm"),
        (1, 64, "BatchNorm"),
        (0, 96, "BatchNorm"),
    ]
    
    successful = 0
    total = len(configurations)
    
    print(f"🎯 Running {total} TST batch experiments...")
    
    for i, (embed_type, dim_model, norm_type) in enumerate(configurations, 1):
        print(f"\n{'='*60}")
        print(f"BATCH EXPERIMENT {i}/{total}")
        print(f"{'='*60}")
        
        if launch_single_tst_experiment(embed_type, dim_model, norm_type):
            successful += 1
        
        # Small delay between experiments
        if i < total:
            print("⏱️ Waiting 30 seconds before next experiment...")
            time.sleep(30)
    
    print(f"\n🎉 Batch experiments completed: {successful}/{total} successful")

def main():
    parser = argparse.ArgumentParser(description="TST Experiments Launcher")
    parser.add_argument("mode", choices=["single", "compare", "batch"], 
                       help="Experiment mode")
    parser.add_argument("--embed-type", type=int, choices=[0, 1], default=0,
                       help="Embedding type (0=None, 1=Temporal)")
    parser.add_argument("--dim-model", type=int, default=64,
                       help="Model dimension")
    parser.add_argument("--norm-type", choices=["BatchNorm", "LayerNorm"], default="BatchNorm",
                       help="Normalization type for TST")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs for comparison")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("RunTransAppPretraining_TST.py").exists():
        print("❌ Error: Please run this script from the experiments_tst directory")
        print("Current directory should contain RunTransAppPretraining_TST.py")
        sys.exit(1)
    
    print("🏗️ TST Experiments Launcher")
    print(f"Mode: {args.mode}")
    print(f"Configuration: embed_type={args.embed_type}, dim_model={args.dim_model}")
    
    if args.mode == "single":
        launch_single_tst_experiment(args.embed_type, args.dim_model, args.norm_type)
    
    elif args.mode == "compare":
        launch_architecture_comparison(args.embed_type, args.dim_model, args.epochs)
    
    elif args.mode == "batch":
        launch_tst_batch_experiments()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        print("🏗️ TST Experiments Launcher - Interactive Mode")
        print("\n📋 Choose experiment type:")
        print("1. Single TST experiment")
        print("2. Architecture comparison (Standard vs TST)")
        print("3. Batch TST experiments")
        print("4. Exit")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
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
                print("\n🚀 Launching batch TST experiments...")
                launch_tst_batch_experiments()
                
            elif choice == "4":
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