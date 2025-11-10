#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia
# @description : appliance detection experiments on CER dataset
# @component: src/utils/
# @file : RunTransAppClassif.py
#
#################################################################################################################

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging
import warnings

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Suppress sklearn FutureWarnings about deprecated functions
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*force_all_finite.*')
warnings.filterwarnings('ignore', message='.*_check_n_features.*')
warnings.filterwarnings('ignore', message='.*_check_feature_names.*')

# Fix path resolution - get to TransApp root directory
current_file = Path(__file__).resolve()
root = current_file.parents[1]  # Go up from experiments/ to TransApp/
sys.path.insert(0, str(root))  # Insert at beginning of path

# Now import with correct paths
try:
    from experiments.data_utils import *
    from src.TransAppModel.TransApp import *
    from src.AD_Framework.Framework import *
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Root directory: {root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

def setup_experiment_logging(case_name, model_name, dim_model, rd_state, save_path):
    """Setup logging for experiment tracking"""
    
    # Create logs directory
    log_dir = Path(save_path).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Setup file logging
    log_filename = f"{case_name}_{model_name}_{dim_model}_seed{rd_state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(f'TransApp_{case_name}_{rd_state}')
    
    return logger, log_filepath

def save_experiment_results(save_path, case_name, model_name, dim_model, rd_state, 
                           dict_params, model_trainer, final_metrics, exo_variable):
    """Save comprehensive experiment results to JSON"""
    
    results_data = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'case_name': case_name,
            'model_name': model_name,
            'dim_model': dim_model,
            'random_seed': rd_state,
            'exo_variables': exo_variable,
            'window_size': 1024
        },
        'hyperparameters': dict_params,
        'training_results': {
            'final_epoch': getattr(model_trainer, 'passed_epochs', None),
            'best_loss': getattr(model_trainer, 'best_loss', None),
            'training_time': getattr(model_trainer, 'train_time', None),
            'evaluation_time': getattr(model_trainer, 'eval_time', None),
            'voter_time': getattr(model_trainer, 'voter_time', None)
        },
        'performance_metrics': final_metrics,
        'model_info': {
            'total_parameters': sum(p.numel() for p in model_trainer.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model_trainer.model.parameters() if p.requires_grad)
        }
    }
    
    # Save to JSON file
    json_filepath = f"{save_path}_results.json"
    with open(json_filepath, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    return json_filepath

def launch_training(model, 
                    save_path, m, win,
                    datas_tuple,
                    dict_params,
                    case_name, model_name, dim_model, rd_state, exo_variable):
    """
    Launch model training with comprehensive logging

    Input :
    - model : model instance
    - save_path : path to save model / case
    - m : number of variable of the MTS
    - win : window size of subsequences
    - datas_tuple : [X_train, y_train, ... X_test_voter , y_test_voter]
    - dict_params : dictionary of parameters
    - case_name, model_name, dim_model, rd_state, exo_variable : for logging
    """
    
    # Setup logging
    logger, log_filepath = setup_experiment_logging(case_name, model_name, dim_model, rd_state, save_path)
    
    logger.info("="*80)
    logger.info(f"STARTING TRANSAPP EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Case: {case_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dimension: {dim_model}")
    logger.info(f"Random Seed: {rd_state}")
    logger.info(f"Exogenous Variables: {exo_variable}")
    logger.info(f"Window Size: {win}")
    logger.info(f"Input Channels: {m}")
    
    logger.info(f"\nHyperparameters:")
    for key, value in dict_params.items():
        logger.info(f"  {key}: {value}")
    
    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Scliced data
    X_train = datas_tuple[0]
    y_train = datas_tuple[1]
    X_valid = datas_tuple[2]
    y_valid = datas_tuple[3]
    X_test  = datas_tuple[4]
    y_test  = datas_tuple[5]

    # Log data shapes
    logger.info(f"\nData Shapes:")
    logger.info(f"  Training: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  Validation: X={X_valid.shape}, y={y_valid.shape}")
    logger.info(f"  Test: X={X_test.shape}, y={y_test.shape}")

    # Entire curves data
    X_train_voter = datas_tuple[6]
    y_train_voter = datas_tuple[7]
    X_valid_voter = datas_tuple[8]
    y_valid_voter = datas_tuple[9]
    X_test_voter  = datas_tuple[10]
    y_test_voter  = datas_tuple[11]
    
    logger.info(f"  Voter Test: X={X_test_voter.shape}, y={y_test_voter.shape}")

    # Dataset
    train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
    valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
    test_dataset  = TSDataset(X_test, y_test,   scaler=True, scale_dim=[0])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dict_params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

    logger.info(f"\nStarting training for {dict_params['epochs']} epochs...")

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
                                 device="cuda", all_gpu=True,
                                 save_checkpoint=True, path_checkpoint=save_path)

    # Training with logging
    training_start_time = datetime.now()
    model_trainer.train(dict_params['epochs'])
    training_end_time = datetime.now()
    
    logger.info(f"\nTraining completed!")
    logger.info(f"  Training time: {model_trainer.train_time}s")
    logger.info(f"  Final epochs: {model_trainer.passed_epochs}")
    logger.info(f"  Best loss: {model_trainer.best_loss:.6f}")

    #============ eval last model ============#
    logger.info(f"\nEvaluating last model...")
    last_model_loss, last_model_metrics = model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1), mask='test_metrics_lastmodel')
    logger.info(f"  Last model metrics: {last_model_metrics}")

    #============ restore best weight and evaluate ============#    
    logger.info(f"\nRestoring best weights and evaluating...")
    model_trainer.restore_best_weights()
    best_model_loss, best_model_metrics = model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))
    logger.info(f"  Best model metrics: {best_model_metrics}")

    #============ find best quantile on valid dataset ============#
    logger.info(f"\nFinding best quantile on validation dataset...")
    best_quantile_metrics = model_trainer.ADFFindBestQuantile(TSDataset(X_valid_voter, y_valid_voter), m=m, win=win)
    logger.info(f"  Best quantile metrics: {best_quantile_metrics}")
    
    logger.info(f"\nEvaluating on test voter dataset...")
    quant_metric = model_trainer.ADFvoter_proba(TSDataset(X_test_voter, y_test_voter), m=m, win=win)
    logger.info(f"  Final voter metrics: {quant_metric}")

    # Save comprehensive results
    json_filepath = save_experiment_results(
        save_path, case_name, model_name, dim_model, rd_state,
        dict_params, model_trainer, quant_metric, exo_variable
    )
    
    logger.info(f"\nExperiment completed!")
    logger.info(f"  Model saved to: {save_path}.pt")
    logger.info(f"  Results saved to: {json_filepath}")
    logger.info(f"  Log saved to: {log_filepath}")
    logger.info("="*80)

    return quant_metric


def get_model_inst(m, win, dim_model, path_select_core=None):

    if path_select_core is not None:
        n_enc_layers = 5 if 'Large' in path_select_core else 3
    else:
        n_enc_layers = 3

    TApp = TransApp(max_len=win, c_in=m,
                    mode="classif",
                    n_embed_blocks=1, 
                    encoding_type='noencoding',
                    n_encoder_layers=n_enc_layers,
                    kernel_size=5,
                    d_model=dim_model, pffn_ratio=2, n_head=4,
                    prenorm=True, norm="LayerNorm",
                    activation='gelu',
                    store_att=False, attn_dp_rate=0.2, head_dp_rate=0., dp_rate=0.2,
                    att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
                    c_reconstruct=1, apply_gap=True, nb_class=2)

    if path_select_core is not None:
        TApp.load_state_dict(torch.load(path_select_core, weights_only=False)['model_state_dict'])

    return TApp


if __name__ == "__main__":
    print("TransApp experiments on CER data detection cases with comprehensive logging.")

    path_results = str(root) + '/results/TransAppResults/'
    path_pretrained_core = str(root) + '/results/TransAppPretrained/'

    case_name  = str(sys.argv[1])
    model_name = str(sys.argv[2])
    dim_model  = int(sys.argv[3])
    frac       = str(sys.argv[4])

    win = 1024

    # List of possible Embedding : Univariate, Time Embedding 
    list_exo_variable = [['hours_cos', 'hours_sin', 'days_cos', 'days_sin']]
    name_exo_variables = ['Embed']

    for i, l in enumerate(list_exo_variable):
        if model_name=='TransApp':
            path_core = None
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                           'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
        elif model_name=='TransAppPT':
            path_core = path_pretrained_core + str(name_exo_variables[i]) + '/TransApp' + str(dim_model) + '.pt'
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 5,
                           'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
        elif model_name=='TransAppLPT':
            path_core = path_pretrained_core + str(name_exo_variables[i]) + '/TransAppL' + str(dim_model) + '_' + frac + '.pt'
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                           'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
        elif model_name=='TransAppLargePT':
            path_core = path_pretrained_core + str(name_exo_variables[i]) + '/TransAppLarge' + str(dim_model) + '_' + frac + '.pt'
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                           'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
        else:
            raise ValueError('Model Name unknown.')

        _ = create_dir(path_results + name_exo_variables[i] + '/')
        path = create_dir(path_results + name_exo_variables[i] + '/' + case_name + '/')

        m = len(l) + 1 # MTS Number of variables

        for rd_state in range(0, 3):
            if model_name=='TransAppLPT' or model_name=='TransAppLargePT':
                save_path = path + model_name + str(dim_model) + '_' + frac + '_' + str(rd_state)
                logging.info(f"Results will be saved in: {save_path}")
            else:
                save_path = path + model_name + str(dim_model) + '_' + str(rd_state)
                logging.info(f"Results will be saved in: {save_path}")

            model = get_model_inst(m=m, win=win, dim_model=dim_model, path_select_core=path_core)

            datas_tuple = CER_get_data_case(case_name, seed=rd_state, exo_variable=l,
                                            win=win, ratio_resample=0.8)
            
            # Launch training with enhanced logging
            launch_training(model, 
                            save_path, m, win, 
                            datas_tuple,  
                            dict_params,
                            case_name, model_name, dim_model, rd_state, l)
