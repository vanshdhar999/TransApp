#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia (Modified for architecture comparison)
# @description : Compare TransApp vs TransApp_TST performance
# @component: experiments_tst/
# @file : CompareArchitectures.py
#
#################################################################################################################

import os, sys
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Fix path resolution - get to TransApp root directory
current_file = Path(__file__).resolve()
root = current_file.parents[1]  # Go up from experiments_tst/ to TransApp/
sys.path.insert(0, str(root))  # Insert at beginning of path

# Now import with correct paths
try:
    from experiments.data_utils import *
    from src.TransAppModel.TransApp import TransApp
    from src.TransAppModel.TransApp_TST import get_transapp_tst_model
    from src.AD_Framework.Framework import *
    from src.utils.losses import *
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Root directory: {root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

def get_model_standard(m, win, dim_model, mode="pretraining"):
    """Get standard TransApp model"""
    return TransApp(
        max_len=win, c_in=m, mode=mode,
        n_embed_blocks=1, encoding_type='noencoding',
        n_encoder_layers=3, kernel_size=5,
        d_model=dim_model, pffn_ratio=2, n_head=4,
        prenorm=True, norm="LayerNorm", activation='gelu',
        store_att=False, attn_dp_rate=0.2, head_dp_rate=0., dp_rate=0.2,
        att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
        c_reconstruct=1, apply_gap=True, nb_class=2
    )

def get_model_tst(m, win, dim_model, mode="pretraining", norm="BatchNorm"):
    """Get TST-enhanced TransApp model"""
    return get_transapp_tst_model(
        m=m, win=win, dim_model=dim_model, mode=mode,
        large_version=False, use_tst_pos_encoding=True,
        norm=norm, res_attention=True
    )

def train_model(model, X_train, model_name, save_path, dict_params):
    """Train a model and return training statistics"""
    
    print(f"\n🏋️ Training {model_name}...")
    start_time = time.time()
    
    # Data preparation
    pretraining_dataset = TSDataset(X_train, scaler=True, scale_dim=[0])
    train_loader = torch.utils.data.DataLoader(pretraining_dataset, 
                                              batch_size=dict_params['batch_size'], 
                                              shuffle=True)
    
    # Geometric mask
    GeomMask = GeometricMask(mean_length=24, masking_ratio=0.5, type_corrupt='zero', dim_masked=0)
    
    # Trainer
    model_pretrainer = self_pretrainer(
        model,
        train_loader, valid_loader=None,
        learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
        name_scheduler='CosineAnnealingLR',
        dict_params_scheduler={'T_max': dict_params['epochs'], 'eta_min': 1e-6},
        warmup_duration=None,
        criterion=MaskedMSELoss(type_loss='L1'), mask=GeomMask,
        device="cuda", all_gpu=False,  # Set to False for comparison
        verbose=True, plotloss=False,
        save_fig=False, path_fig=None,
        save_only_core=False,
        save_checkpoint=True, path_checkpoint=save_path
    )
    
    # Training
    model_pretrainer.train(dict_params['epochs'])
    
    training_time = time.time() - start_time
    
    # Get final loss if available
    final_loss = getattr(model_pretrainer, 'train_loss', [0])[-1] if hasattr(model_pretrainer, 'train_loss') and model_pretrainer.train_loss else None
    
    return {
        'model_name': model_name,
        'training_time': training_time,
        'final_loss': final_loss,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

def compare_architectures(embed_type=0, dim_model=64, epochs=10):
    """Compare standard TransApp vs TST-enhanced TransApp"""
    
    print(f"\n{'='*80}")
    print(f"ARCHITECTURE COMPARISON")
    print(f"Configuration: embed_type={embed_type}, dim_model={dim_model}, epochs={epochs}")
    print(f"{'='*80}")
    
    # Configuration
    win = 1024
    list_exo_variable = [[], ['hours_cos', 'hours_sin', 'days_cos', 'days_sin']]
    name_exo_variables = ['None', 'Embed']
    
    m = len(list_exo_variable[embed_type]) + 1
    
    # Create results directories
    comparison_dir = str(root) + '/results/Architecture_Comparison/'
    _ = create_dir(comparison_dir)
    
    standard_dir = comparison_dir + 'Standard/'
    tst_dir = comparison_dir + 'TST/'
    _ = create_dir(standard_dir)
    _ = create_dir(tst_dir)
    
    # Training parameters
    dict_params = {'lr': 1e-4, 'wd': 1e-4, 'batch_size': 16, 'epochs': epochs}
    
    # Load data
    print(f"📊 Loading pretraining data with {m} channels...")
    X_train = CER_get_data_pretraining(exo_variable=list_exo_variable[embed_type])
    
    # Model paths
    standard_path = standard_dir + f'TransApp_{dim_model}_embed{embed_type}'
    tst_batchnorm_path = tst_dir + f'TransApp_TST_{dim_model}_BatchNorm_embed{embed_type}'
    tst_layernorm_path = tst_dir + f'TransApp_TST_{dim_model}_LayerNorm_embed{embed_type}'
    
    results = []
    
    print(f"\n🔵 Creating Standard TransApp model...")
    model_standard = get_model_standard(m, win, dim_model, mode="pretraining")
    print(f"   Parameters: {sum(p.numel() for p in model_standard.parameters()):,}")
    
    result_standard = train_model(model_standard, X_train, "Standard TransApp", 
                                 standard_path, dict_params)
    results.append(result_standard)
    
    # 2. Train TST with BatchNorm
    print(f"\n🟡 Creating TST TransApp model (BatchNorm)...")
    model_tst_bn = get_model_tst(m, win, dim_model, mode="pretraining", norm="BatchNorm")
    print(f"   Parameters: {sum(p.numel() for p in model_tst_bn.parameters()):,}")
    
    result_tst_bn = train_model(model_tst_bn, X_train, "TST TransApp (BatchNorm)", 
                               tst_batchnorm_path, dict_params)
    results.append(result_tst_bn)
    
    # 3. Train TST with LayerNorm
    print(f"\n🟢 Creating TST TransApp model (LayerNorm)...")
    model_tst_ln = get_model_tst(m, win, dim_model, mode="pretraining", norm="LayerNorm")
    print(f"   Parameters: {sum(p.numel() for p in model_tst_ln.parameters()):,}")
    
    result_tst_ln = train_model(model_tst_ln, X_train, "TST TransApp (LayerNorm)", 
                               tst_layernorm_path, dict_params)
    results.append(result_tst_ln)
    
    # Results summary
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*80}")
    
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'embed_type': embed_type,
            'dim_model': dim_model,
            'epochs': epochs,
            'exo_variables': list_exo_variable[embed_type]
        },
        'results': results
    }
    
    # Print comparison table
    print(f"\n📊 Performance Comparison:")
    print(f"{'Model':<25} {'Parameters':<12} {'Time (s)':<10} {'Final Loss':<12}")
    print("-" * 65)
    
    for result in results:
        model_name = result['model_name']
        params = f"{result['total_params']:,}"
        train_time = f"{result['training_time']:.1f}"
        final_loss = f"{result['final_loss']:.6f}" if result['final_loss'] is not None else "N/A"
        
        print(f"{model_name:<25} {params:<12} {train_time:<10} {final_loss:<12}")
    
    # Save comparison results
    comparison_file = comparison_dir + f'comparison_embed{embed_type}_dim{dim_model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\n💾 Comparison results saved to: {comparison_file}")
    
    # Efficiency analysis
    print(f"\n📈 Efficiency Analysis:")
    standard_result = results[0]
    
    for i, result in enumerate(results[1:], 1):
        time_ratio = result['training_time'] / standard_result['training_time']
        param_ratio = result['total_params'] / standard_result['total_params']
        
        print(f"   {result['model_name']} vs Standard:")
        print(f"     Parameter ratio: {param_ratio:.2f}x")
        print(f"     Training time ratio: {time_ratio:.2f}x")
        
        if result['final_loss'] is not None and standard_result['final_loss'] is not None:
            loss_improvement = ((standard_result['final_loss'] - result['final_loss']) / standard_result['final_loss']) * 100
            print(f"     Loss improvement: {loss_improvement:+.2f}%")
    
    return comparison_data

def run_comprehensive_comparison():
    """Run comprehensive comparison across multiple configurations"""
    
    print("🚀 Running comprehensive architecture comparison...")
    
    configurations = [
        {'embed_type': 0, 'dim_model': 64, 'epochs': 10},
        {'embed_type': 1, 'dim_model': 64, 'epochs': 10},
        {'embed_type': 0, 'dim_model': 96, 'epochs': 10},
    ]
    
    all_results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE COMPARISON {i}/{len(configurations)}")
        print(f"{'='*80}")
        
        try:
            result = compare_architectures(**config)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error in configuration {i}: {e}")
            continue
        
        # Small break between configurations
        time.sleep(2)
    
    # Save comprehensive results
    comprehensive_file = str(root) + f'/results/Architecture_Comparison/comprehensive_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(comprehensive_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n🎉 Comprehensive comparison completed!")
    print(f"💾 Results saved to: {comprehensive_file}")
    
    return all_results

if __name__ == "__main__":
    print("🏗️ TransApp Architecture Comparison Suite")
    
    if len(sys.argv) >= 3:
        # Command line mode
        try:
            embed_type = int(sys.argv[1])
            dim_model = int(sys.argv[2])
            epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            
            compare_architectures(embed_type, dim_model, epochs)
            
        except ValueError as e:
            print(f"❌ Error: Invalid arguments. {e}")
            print("Usage: python CompareArchitectures.py <embed_type> <dim_model> [epochs]")
            sys.exit(1)
    
    else:
        # Interactive mode
        print("\n📋 Choose comparison mode:")
        print("1. Single configuration comparison")
        print("2. Comprehensive comparison (multiple configurations)")
        print("3. Exit")
        
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                embed_type = int(input("Enter embed_type (0=None, 1=Temporal): "))
                dim_model = int(input("Enter model dimension (e.g., 64, 96): "))
                epochs = int(input("Enter number of epochs [10]: ") or "10")
                
                compare_architectures(embed_type, dim_model, epochs)
                
            elif choice == "2":
                run_comprehensive_comparison()
                
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