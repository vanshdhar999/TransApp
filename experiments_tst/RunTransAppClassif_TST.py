#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia (Modified for TST architecture)
# @description : TST-Enhanced TransApp appliance detection experiments
# @component: experiments_tst/
# @file : RunTransAppClassif_TST.py
#
#################################################################################################################

import os, sys
import numpy as np
import pandas as pd
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
    from src.TransAppModel.TransApp_TST import *
    from src.AD_Framework.Framework import *
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Root directory: {root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

def clean_gpu_memory():
    """Clean GPU memory cache and collect garbage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()

def launch_training_tst(model, 
                        save_path, m, win,
                        datas_tuple,
                        dict_params,
                        case_name):
    """
    Launch TST model training following the same methodology as original TransApp

    Input :
    - model : TST model instance
    - save_path : path to save model / case
    - m : number of variable of the MTS
    - win : window size of subsequences
    - datas_tuple : [X_train, y_train, ... X_test_voter , y_test_voter]
    - dict_params : dictionary of parameters
    - case_name : name of the detection case
    """
    
    print(f"🎯 Starting TST fine-tuning for {case_name}...")

    # Sliced data for training
    X_train = datas_tuple[0]
    y_train = datas_tuple[1]
    X_valid = datas_tuple[2]
    y_valid = datas_tuple[3]
    X_test  = datas_tuple[4]
    y_test  = datas_tuple[5]

    # Entire curves data for evaluation
    X_train_voter = datas_tuple[6]
    y_train_voter = datas_tuple[7]
    X_valid_voter = datas_tuple[8]
    y_valid_voter = datas_tuple[9]
    X_test_voter  = datas_tuple[10]
    y_test_voter  = datas_tuple[11]

    print(f"📊 Data shapes:")
    print(f"   Training: {X_train.shape}, {y_train.shape}")
    print(f"   Validation: {X_valid.shape}, {y_valid.shape}")
    print(f"   Test: {X_test.shape}, {y_test.shape}")
    print(f"   Voter test: {X_test_voter.shape}, {y_test_voter.shape}")

    # Dataset preparation (same as original)
    train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
    valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
    test_dataset  = TSDataset(X_test, y_test,   scaler=True, scale_dim=[0])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dict_params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

    # AD Framework trainer (same as original methodology)
    model_trainer = AD_Framework(model,
                                 train_loader=train_loader, valid_loader=valid_loader,
                                 learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
                                 criterion=nn.CrossEntropyLoss(),
                                 patience_es=dict_params['p_es'], patience_rlr=dict_params['p_rlr'],
                                 f_metrics=getmetrics(),
                                 n_warmup_epochs=dict_params['n_warmup_epochs'],
                                 scale_by_subseq_in_voter=True, scale_dim=[0],
                                 verbose=True, plotloss=False, 
                                 save_fig=False, path_fig=None,
                                 device="cuda", all_gpu=False,  # Set to False for stability
                                 save_checkpoint=True, path_checkpoint=save_path)

    print(f"🏋️ Training TST model for {dict_params['epochs']} epochs...")
    model_trainer.train(dict_params['epochs'])

    print("📊 Evaluating TST model...")
    
    #============ eval last model on subsequences ============#
    model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1), mask='test_metrics_lastmodel')

    #============ restore best weight and evaluate ============#    
    model_trainer.restore_best_weights()
    subseq_metrics = model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))

    #============ find best quantile on valid voter dataset ============#
    print("🔍 Finding best quantile on validation voter dataset...")
    model_trainer.ADFFindBestQuantile(TSDataset(X_valid_voter, y_valid_voter), m=m, win=win)
    
    #============ evaluate on test voter dataset using best quantile ============#
    print("📈 Evaluating on test voter dataset...")
    quant_metric = model_trainer.ADFvoter_proba(TSDataset(X_test_voter, y_test_voter), m=m, win=win)
    print(f"Quantile metrics: {quant_metric}")

    # Clean GPU memory after training
    clean_gpu_memory()
    
    return {
        'subsequence_metrics': subseq_metrics,
        'quantile_metrics': quant_metric,
        'best_quantile': getattr(model_trainer, 'best_quantile', None)
    }

def get_model_inst_tst(m, win, dim_model, norm_type="BatchNorm", path_select_core=None):
    """
    Get TST-enhanced TransApp model instance for classification
    
    Parameters:
        m: int - number of channels of input time series
        win: int - length of input subsequence
        dim_model: int - model dimension
        norm_type: str - normalization type ("BatchNorm" or "LayerNorm")
        path_select_core: str - path to pretrained TST instance
    """
    
    # Determine if this is a large version based on the path
    large_version = path_select_core is not None and 'Large' in path_select_core
    
    # Create TST model in classification mode
    TApp = get_transapp_tst_model(
        m=m, win=win, dim_model=dim_model,
        mode="classif",  # Classification mode
        large_version=large_version,
        use_tst_pos_encoding=True,
        norm=norm_type,
        res_attention=True
    )

    # Load pretrained weights if provided
    if path_select_core is not None:
        try:
            checkpoint = torch.load(path_select_core, weights_only=False)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load with strict=False to handle mode changes (pretraining -> classif)
            missing_keys, unexpected_keys = TApp.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys when loading pretrained weights: {unexpected_keys}")
                
            print(f"✅ Loaded pretrained TST weights from: {path_select_core}")
            
        except Exception as e:
            print(f"❌ Error loading pretrained weights: {e}")
            print("   Continuing with random initialization...")

    return TApp

def get_available_cases():
    """Get list of available detection cases"""
    cer_cases = [
        'cooker_case', 'dishwasher_case', 'waterheater_case', 'pluginheater_case',
        'tumbledryer_case', 'tv_greater21inch_case', 'tv_lessr21inch_case',
        'desktopcomputer_case', 'laptopcomputer_case'
    ]
    
    comstock_cases = [
        'cooling_case', 'heating_case', 'waterheating_case'
    ]
    
    return {
        'CER': cer_cases,
        'COMSTOCK': comstock_cases
    }

def save_experiment_results(results, save_path, case_name, model_name, dim_model, config):
    """Save experiment results to JSON file"""
    
    experiment_data = {
        'timestamp': datetime.now().isoformat(),
        'case_name': case_name,
        'model_name': model_name,
        'dim_model': dim_model,
        'configuration': config,
        'results': results
    }
    
    results_file = save_path + '_results.json'
    with open(results_file, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {results_file}")
    return results_file

if __name__ == "__main__":
    print("🚀 TST-Enhanced TransApp experiments on appliance detection cases")
    
    # Check command line arguments
    if len(sys.argv) < 5:
        print("\n❌ Error: Insufficient arguments")
        print("Usage: python RunTransAppClassif_TST.py <case_name> <model_name> <dim_model> <epochs> [norm_type]")
        print("\nAvailable cases:")
        cases = get_available_cases()
        for dataset, case_list in cases.items():
            print(f"  {dataset}: {', '.join(case_list)}")
        print("\nModel types:")
        print("  - TransApp_TST: TST from scratch")
        print("  - TransApp_TST_PT: TST with CER pretraining")
        print("  - TransApp_TST_COMSTOCK_PT: TST with COMSTOCK pretraining")
        print("\nExample: python RunTransAppClassif_TST.py cooker_case TransApp_TST_PT 64 15 BatchNorm")
        sys.exit(1)

    # Parse arguments
    case_name = str(sys.argv[1])
    model_name = str(sys.argv[2])
    dim_model = int(sys.argv[3])
    epochs = int(sys.argv[4])
    norm_type = str(sys.argv[5]) if len(sys.argv) > 5 else "BatchNorm"
    
    # Validate arguments
    available_cases = get_available_cases()
    all_cases = []
    for case_list in available_cases.values():
        all_cases.extend(case_list)
    
    if case_name not in all_cases:
        print(f"❌ Error: Unknown case '{case_name}'")
        print(f"Available cases: {all_cases}")
        sys.exit(1)
    
    if norm_type not in ["BatchNorm", "LayerNorm"]:
        print(f"❌ Error: norm_type must be 'BatchNorm' or 'LayerNorm', got '{norm_type}'")
        sys.exit(1)

    # Determine dataset type
    dataset_type = None
    for ds_type, case_list in available_cases.items():
        if case_name in case_list:
            dataset_type = ds_type
            break

    print(f"\n📋 Configuration:")
    print(f"   Case: {case_name} ({dataset_type})")
    print(f"   Normalization: {norm_type}")

    # Set up paths and parameters
    path_results = str(root) + '/results/TransAppResults_TST/'
    path_pretrained_core = str(root) + '/results/TransAppPretrained_TST/'

    win = 128  # Fine-tuning window size (same as original)

    # List of possible embeddings
    list_exo_variable = [[], ['hours_cos', 'hours_sin', 'days_cos', 'days_sin']]
    name_exo_variables = ['None', 'Embed']

    # Run experiments for each embedding type
    all_results = []
    
    for i, exo_vars in enumerate(list_exo_variable):
        embed_name = name_exo_variables[i]
        m = len(exo_vars) + 1  # Number of variables
        
        print(f"\n{'='*70}")
        print(f"RUNNING EXPERIMENT: {embed_name} embedding")
        print(f"{'='*70}")
        
        # Configure model and parameters based on model_name
        if model_name == 'TransApp_TST':
            # TST from scratch
            path_core = None
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': epochs,
                          'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
            
        elif model_name == 'TransApp_TST_PT':
            # TST with CER pretraining
            path_core = path_pretrained_core + f'{embed_name}/TransApp_TST_{dim_model}_{norm_type}.pt'
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': epochs,
                          'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
            
        elif model_name == 'TransApp_TST_COMSTOCK_PT':
            # TST with COMSTOCK pretraining
            path_core = path_pretrained_core + f'{embed_name}/TransApp_TST_COMSTOCK_{dim_model}_{norm_type}.pt'
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': epochs,
                          'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
            
        else:
            print(f"❌ Error: Unknown model name '{model_name}'")
            print("Available models: TransApp_TST, TransApp_TST_PT, TransApp_TST_COMSTOCK_PT")
            sys.exit(1)

        # Create results directory
        result_dir = path_results + f'{embed_name}/'
        _ = create_dir(result_dir)
        case_dir = create_dir(result_dir + f'{case_name}/')

        # Run multiple random seeds (same as original)
        for rd_state in range(0, 3):
            print(f"\n🎲 Running random seed {rd_state}...")
            
            # Create save path
            save_path = case_dir + f'{model_name}_{dim_model}_{norm_type}_{epochs}ep_{rd_state}'

            # Create model instance
            model = get_model_inst_tst(m=m, win=win, dim_model=dim_model, 
                                     norm_type=norm_type, path_select_core=path_core)

            # Get data based on dataset type
            if dataset_type == 'CER':
                datas_tuple = CER_get_data_case(case_name, seed=rd_state, exo_variable=exo_vars,
                                              win=win, ratio_resample=0.8)
            elif dataset_type == 'COMSTOCK':
                datas_tuple = COMSTOCK_get_data_case(case_name, seed=rd_state)
            else:
                print(f"❌ Error: Unknown dataset type '{dataset_type}'")
                continue

            # Launch training
            try:
                results = launch_training_tst(model, save_path, m, win, datas_tuple, dict_params, case_name)
                
                # Save results
                config = {
                    'embed_type': i,
                    'exo_variables': exo_vars,
                    'random_seed': rd_state,
                    'dataset_type': dataset_type,
                    'norm_type': norm_type,
                    'parameters': dict_params
                }
                
                save_experiment_results(results, save_path, case_name, model_name, dim_model, config)
                all_results.append({
                    'embed_name': embed_name,
                    'random_seed': rd_state,
                    'results': results,
                    'config': config
                })
                
                print(f"✅ Completed seed {rd_state} for {embed_name} embedding")
                
            except Exception as e:
                print(f"❌ Error in seed {rd_state} for {embed_name} embedding: {e}")
                continue

    # Save comprehensive results
    if all_results:
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'experiment_info': {
                'case_name': case_name,
                'model_name': model_name,
                'dim_model': dim_model,
                'epochs': epochs,
                'norm_type': norm_type,
                'dataset_type': dataset_type
            },
            'all_results': all_results
        }
        
        comprehensive_file = path_results + f'comprehensive_{case_name}_{model_name}_{dim_model}_{norm_type}_{epochs}ep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(comprehensive_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n🎉 All experiments completed!")
        print(f"💾 Comprehensive results saved to: {comprehensive_file}")
        
        # Print summary
        print(f"\n📊 Summary:")
        for result in all_results:
            embed_name = result['embed_name']
            seed = result['random_seed']
            quant_metrics = result['results'].get('quantile_metrics', {})
            
            if quant_metrics:
                f1_score = quant_metrics.get('f1_score', 'N/A')
                accuracy = quant_metrics.get('accuracy', 'N/A')
                print(f"   {embed_name} (seed {seed}): F1={f1_score}, Acc={accuracy}")
    else:
        print("❌ No experiments completed successfully")
