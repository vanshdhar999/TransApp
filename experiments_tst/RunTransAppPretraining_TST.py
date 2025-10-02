#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia (Modified for TST testing)
# @description : TST-Enhanced TransApp Pretraining
# @component: experiments_tst/
# @file : RunTransAppPretraining_TST.py
#
#################################################################################################################

import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from pathlib import Path
root = Path(os.getcwd()).resolve().parents[0]
sys.path.append(str(root))
from experiments.data_utils import *
from src.TransAppModel.TransApp_TST import *
from src.AD_Framework.Framework import *
from src.utils.losses import *

def launch_pretraining(model, 
                       save_path, m, win,
                       X_train,
                       GeomMask,
                       dict_params):

    pretraining_dataset = TSDataset(X_train, scaler=True, scale_dim=[0])
    train_loader = torch.utils.data.DataLoader(pretraining_dataset, batch_size=dict_params['batch_size'], shuffle=True)

    model_pretrainer = self_pretrainer(model,                                     
                                       train_loader, valid_loader=None,
                                       learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
                                       name_scheduler='CosineAnnealingLR',
                                       dict_params_scheduler={'T_max': dict_params['epochs'], 'eta_min': 1e-6},
                                       warmup_duration=None,
                                       criterion=MaskedMSELoss(type_loss='L1'), mask=GeomMask,
                                       device="cuda", all_gpu=True,
                                       verbose=True, plotloss=False, 
                                       save_fig=False, path_fig=None,
                                       save_only_core=False,
                                       save_checkpoint=True, path_checkpoint=save_path)

    model_pretrainer.train(dict_params['epochs'])

    return

def get_model_inst_tst(m, win, dim_model, use_tst_pos=True, norm="BatchNorm"):
    """
    Get TST-enhanced TransApp model instance
    """
    TApp = get_transapp_tst_model(
        m=m, win=win, dim_model=dim_model, 
        mode="pretraining",
        large_version=False,
        use_tst_pos_encoding=use_tst_pos,
        norm=norm,
        res_attention=True
    )
    
    return TApp

def print_usage():
    print("Usage: python RunTransAppPretraining_TST.py <embed_type> <dim_model> [norm_type]")
    print("  embed_type: 0 (no exogenous vars) or 1 (with temporal encoding)")
    print("  dim_model: model dimension (e.g., 64, 96, 128)")
    print("  norm_type: 'BatchNorm' (default) or 'LayerNorm'")
    print("Example: python RunTransAppPretraining_TST.py 0 64 BatchNorm")

def run_single_experiment(embed_type, dim_model, norm_type="BatchNorm"):
    """Run a single TST pretraining experiment"""
    
    print(f"\n🚀 Starting TST experiment: embed_type={embed_type}, dim_model={dim_model}, norm={norm_type}")
    
    win = 1024

    list_exo_variable = [[], ['hours_cos', 'hours_sin', 'days_cos', 'days_sin']]
    name_exo_variables = ['None', 'Embed']

    # Create TST-specific results directory
    path_results = str(root) + '/results/TransAppPretrained_TST/' + name_exo_variables[embed_type] + '/'
    _ = create_dir(path_results)

    dict_params = {'lr': 1e-4, 'wd': 1e-4, 'batch_size': 16, 'epochs': 20}

    path = path_results + 'TransApp_TST_' + str(dim_model) + '_' + norm_type
    m = len(list_exo_variable[embed_type]) + 1

    print(f"📊 Loading pretraining data with {m} channels...")
    X_train = CER_get_data_pretraining(exo_variable=list_exo_variable[embed_type])

    print(f"🏗️ Creating TST-enhanced model with dimension {dim_model} and {norm_type}...")
    model = get_model_inst_tst(m, win, dim_model=dim_model, norm=norm_type)

    # Print model summary
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"   Could not compute model statistics: {e}")

    GeomMask = GeometricMask(mean_length=24, masking_ratio=0.5, type_corrupt='zero', dim_masked=0)

    print(f"💾 Model will be saved to: {path}")
    
    launch_pretraining(model, path, m, win, X_train, GeomMask, dict_params)
    
    print(f"✅ Completed TST experiment: embed_type={embed_type}, dim_model={dim_model}, norm={norm_type}")

def run_comparison_experiments():
    """Run comparison between different TST configurations"""
    
    embed_types = [0, 1]
    dim_models = [64, 96]  # Start with smaller set for comparison
    norm_types = ["BatchNorm", "LayerNorm"]
    
    total_experiments = len(embed_types) * len(dim_models) * len(norm_types)
    current_experiment = 0
    
    print(f"🎯 Running {total_experiments} TST comparison experiments")
    
    for embed_type in embed_types:
        for dim_model in dim_models:
            for norm_type in norm_types:
                current_experiment += 1
                print(f"\n{'='*70}")
                print(f"TST EXPERIMENT {current_experiment}/{total_experiments}")
                print(f"{'='*70}")
                
                try:
                    run_single_experiment(embed_type, dim_model, norm_type)
                except Exception as e:
                    print(f"❌ Error in experiment {current_experiment}: {e}")
                    continue
    
    print(f"\n🎉 All TST comparison experiments completed!")

if __name__ == "__main__":
    print("🚀 TransApp with TST Architecture - Pretraining Suite")
    
    # Check if arguments are provided
    if len(sys.argv) >= 3:
        try:
            embed_type = int(sys.argv[1])
            dim_model = int(sys.argv[2])
            norm_type = sys.argv[3] if len(sys.argv) > 3 else "BatchNorm"
            
            # Validate arguments
            if embed_type not in [0, 1]:
                print(f"❌ Error: embed_type must be 0 or 1, got {embed_type}")
                print_usage()
                sys.exit(1)
                
            if norm_type not in ["BatchNorm", "LayerNorm"]:
                print(f"❌ Error: norm_type must be 'BatchNorm' or 'LayerNorm', got {norm_type}")
                print_usage()
                sys.exit(1)
                
            run_single_experiment(embed_type, dim_model, norm_type)
            
        except ValueError as e:
            print(f"❌ Error: Invalid arguments. {e}")
            print_usage()
            sys.exit(1)
    
    # Interactive mode
    elif len(sys.argv) == 1:
        print("\n📋 Choose mode:")
        print("1. Run single TST experiment")
        print("2. Run TST comparison experiments (BatchNorm vs LayerNorm)")
        print("3. Exit")
        
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                print("\n📋 Single TST Experiment Configuration:")
                embed_type = int(input("Enter embed_type (0=None, 1=Temporal): "))
                dim_model = int(input("Enter model dimension (e.g., 64, 96, 128): "))
                norm_type = input("Enter norm type (BatchNorm/LayerNorm) [BatchNorm]: ").strip() or "BatchNorm"
                
                if embed_type not in [0, 1]:
                    print("❌ Error: embed_type must be 0 or 1")
                    sys.exit(1)
                    
                if norm_type not in ["BatchNorm", "LayerNorm"]:
                    print("❌ Error: norm_type must be 'BatchNorm' or 'LayerNorm'")
                    sys.exit(1)
                    
                run_single_experiment(embed_type, dim_model, norm_type)
                
            elif choice == "2":
                print("\n🚀 Running TST comparison experiments...")
                run_comparison_experiments()
                
            elif choice == "3":
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
        print("❌ Error: Invalid number of arguments")
        print_usage()
        sys.exit(1)