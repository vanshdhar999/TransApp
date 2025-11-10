#################################################################################################################
#
# @copyright : ©2023 EDF  
# @author : Adrien Petralia (Modified for Multi-Stage TST experiments)
# @description : Multi-Stage TST experiments with ComStock pretraining and CER fine-tuning
# @component: experiments_tst/
# @file : RunMultiStageTST.py
#
#################################################################################################################

import os, sys
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Fix path resolution - get to TransApp root directory
current_file = Path(__file__).resolve()
root = current_file.parents[1]  # Go up from experiments_tst/ to TransApp/
sys.path.insert(0, str(root))  # Insert at beginning of path

# Now import with correct paths
try:
    from experiments.data_utils import *
    from src.TransAppModel.TransApp_TST import *
    from src.AD_Framework.Framework import *
    from src.utils.losses import *
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Root directory: {root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

class MultiStageTSTExperiment:
    """
    Multi-stage TST experiment following the recommended setup:
    1. Pretrain on ComStock (30min) with MAE
    2. Optional fine-tune on ComStock with multi-label classification  
    3. Final fine-tune on CER for appliance ownership classification
    """
    
    def __init__(self, experiment_name="multistage_tst", 
                 win=1024, dim_model=128, device='auto'):
        self.experiment_name = experiment_name
        self.win = win
        self.dim_model = dim_model
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🚀 Multi-Stage TST Experiment: {experiment_name}")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Window size: {win}, Model dimension: {dim_model}")
        
        # Create results directory
        self.results_dir = Path(root) / "results" / "MultiStageTST"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Track experiment results
        self.stage_results = {}
        
    def stage1_comstock_pretraining(self, resolution='30min', epochs=15, 
                                   lr=1e-4, batch_size=64, mask_ratio=0.4):
        """
        Stage 1: Self-supervised pretraining on ComStock data using Masked Autoencoder (MAE)
        """
        print("\n" + "="*60)
        print("🎯 STAGE 1: ComStock MAE Pretraining")
        print("="*60)
        
        # Load ComStock data for pretraining (no labels needed)
        print(f"📂 Loading ComStock {resolution} data for pretraining...")
        X_train = COMSTOCK_get_data_pretraining(resolution=resolution, 
                                               seed=0, 
                                               win=self.win,
                                               entire_curve_normalization=True)
        
        print(f"✅ Loaded pretraining data: {X_train.shape}")
        
        # Create model for pretraining (m=1 for ComStock, no exogenous variables)
        m = 1
        model = get_transapp_tst_model(m=m, 
                                      win=self.win, 
                                      dim_model=self.dim_model,
                                      mode="pretraining",
                                      use_tst_pos_encoding=True,  # Use positional encoding
                                      res_attention=True)
        model = model.to(self.device)
        
        print(f"🏗️ Created TST model for pretraining:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create MAE pretraining dataset with masking
        pretraining_dataset = MAETSDataset(X_train, mask_ratio=mask_ratio, scaler=True, scale_dim=[0])
        train_loader = torch.utils.data.DataLoader(pretraining_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True)
        
        # Pretraining parameters
        dict_params = {
            'lr': lr,
            'wd': 1e-5,
            'batch_size': batch_size,
            'n_epochs': epochs,
            'early_stopping': True,
            'patience': 10,
            'reduce_lr': True,
            'factor': 0.5,
            'patience_reduce': 5
        }
        
        # Run pretraining
        print(f"🔥 Starting MAE pretraining for {epochs} epochs...")
        save_path = self.results_dir / f"{self.experiment_name}_stage1_comstock_pretrained.pt"
        
        model_pretrainer = MAEPretrainer(model, 
                                        train_loader, 
                                        valid_loader=None,
                                        learning_rate=dict_params['lr'], 
                                        weight_decay=dict_params['wd'],
                                        name_scheduler='CosineAnnealingLR',
                                        n_epochs=dict_params['n_epochs'],
                                        early_stopping=dict_params['early_stopping'],
                                        patience=dict_params['patience'],
                                        reduce_lr=dict_params['reduce_lr'],
                                        factor=dict_params['factor'],
                                        patience_reduce=dict_params['patience_reduce'],
                                        verbose=True,
                                        device=self.device)
        
        # Train and save
        train_losses, val_losses = model_pretrainer.fit()
        
        # Save pretrained model
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'hyperparameters': dict_params,
            'model_config': {
                'm': m,
                'win': self.win,
                'dim_model': self.dim_model,
                'resolution': resolution
            }
        }, save_path)
        
        print(f"💾 Saved pretrained model to: {save_path}")
        
        # Store results
        self.stage_results['stage1_pretraining'] = {
            'dataset': f'ComStock_{resolution}',
            'task': 'MAE_pretraining',
            'epochs': epochs,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'model_path': str(save_path),
            'data_shape': X_train.shape
        }
        
        return model, save_path
    
    def stage2_comstock_finetuning(self, pretrained_model_path, resolution='30min',
                                  case_names=['cooling_case', 'heating_case', 'water_systems_case'],
                                  epochs=15, lr=1e-5, batch_size=32):
        """
        Stage 2: Optional fine-tuning on ComStock with multi-label ON/OFF classification
        """
        print("\n" + "="*60)
        print("🎯 STAGE 2: ComStock Multi-Label Fine-tuning")
        print("="*60)
        
        # Load pretrained model
        print(f"📂 Loading pretrained model from: {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
        
        # Create model for classification
        m = 1  # ComStock has no exogenous variables
        model = get_transapp_tst_model(m=m, 
                                      win=self.win, 
                                      dim_model=self.dim_model,
                                      mode="classif",
                                      nb_class=len(case_names),  # Multi-label output
                                      use_tst_pos_encoding=True,
                                      res_attention=True)
        
        # Load pretrained weights (encoder part only, skip classification head)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filter out classification head layers that have size mismatches
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"⚠️ Skipping layer {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
            else:
                print(f"⚠️ Skipping layer {k}: not found in current model")
        
        # Update model with compatible weights
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        model = model.to(self.device)
        
        print(f"✅ Loaded {len(filtered_dict)} compatible layers from pretrained model")
        print(f"🏗️ Model setup for {len(case_names)} appliances: {case_names}")
        
        # Load ComStock data for multiple appliances
        print(f"📂 Loading ComStock {resolution} data for multi-label classification...")
        print(f"💡 Available ComStock appliance cases:")
        print(f"   {['cooling_case', 'exterior_lighting_case', 'fans_case', 'heat_recovery_case', 'heat_rejection_case', 'heating_case', 'interior_equipment_case', 'interior_lighting_case', 'pumps_case', 'refrigeration_case', 'water_systems_case']}")
        print(f"🎯 Attempting to load: {case_names}")
        
        # Collect data for all appliances
        all_data = []
        all_labels = []
        successfully_loaded_cases = []  # Track which cases were successfully loaded
        
        for case_name in case_names:
            try:
                data_tuple = COMSTOCK_get_data_case(
                    case_name=case_name,
                    resolution=resolution,
                    seed=0,
                    win=self.win,

                    ratio_resample=0.8
                )
                
                # Unpack the 12 returned values (including voter data)
                (X_train, y_train, X_valid, y_valid, X_test, y_test, 
                 X_train_voter, y_train_voter, X_valid_voter, y_valid_voter, 
                 X_test_voter, y_test_voter) = data_tuple
                
                # Store data and labels
                if len(all_data) == 0:
                    all_data = [X_train, X_valid, X_test]
                    # Handle both numpy arrays and pandas Series/DataFrames
                    if hasattr(y_train, 'values'):
                        all_labels = [y_train.values.reshape(-1, 1), 
                                     y_valid.values.reshape(-1, 1), 
                                     y_test.values.reshape(-1, 1)]
                    else:
                        all_labels = [y_train.reshape(-1, 1), 
                                     y_valid.reshape(-1, 1), 
                                     y_test.reshape(-1, 1)]
                else:
                    # Concatenate labels (assuming same houses across appliances)
                    for i, (new_y, old_y) in enumerate(zip([y_train, y_valid, y_test], all_labels)):
                        if hasattr(new_y, 'values'):
                            new_y_array = new_y.values.reshape(-1, 1)
                        else:
                            new_y_array = new_y.reshape(-1, 1)
                        all_labels[i] = np.concatenate([old_y, new_y_array], axis=1)
                        
                print(f"  ✅ Loaded {case_name}: {y_train.sum()} positive samples")
                successfully_loaded_cases.append(case_name)  # Track successful loads
                
            except Exception as e:
                print(f"  ❌ Failed to load {case_name}: {e}")
                # Continue to try other appliances but don't add to successful list
                continue
                
        # Check if any data was loaded
        if len(all_data) == 0 or len(all_labels) == 0:
            raise ValueError(f"❌ Failed to load any appliance data. No valid case names found. "
                           f"Available ComStock cases: {case_names}")
        
        # If we only have one appliance, convert to single-label classification
        if len(successfully_loaded_cases) == 1:
            print(f"⚠️ Only one appliance loaded, converting to single-label classification")
            # Convert to binary classification instead of multi-label
            actual_nb_classes = 2  # Binary classification
            
            # Recreate model with correct number of classes
            model = get_transapp_tst_model(m=1, 
                                          win=self.win, 
                                          dim_model=self.dim_model,
                                          mode="classif",
                                          nb_class=actual_nb_classes,  # Binary classification
                                          use_tst_pos_encoding=True,
                                          res_attention=True)
            
            # Reload pretrained weights with updated model
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            
            # Filter out classification head layers that have size mismatches
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        print(f"⚠️ Skipping layer {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            model = model.to(self.device)
            print(f"✅ Reloaded {len(filtered_dict)} compatible layers for binary classification")
            
            # Convert labels to binary format (flatten single-label format)
            X_train, X_valid, X_test = all_data
            y_train, y_valid, y_test = all_labels
            y_train = y_train.flatten()  # Convert from (N,1) to (N,)
            y_valid = y_valid.flatten()
            y_test = y_test.flatten()
            
            print(f"📊 Binary classification data shape:")
            print(f"  Train: X={X_train.shape}, y={y_train.shape}")
            print(f"  Valid: X={X_valid.shape}, y={y_valid.shape}")
            print(f"  Test: X={X_test.shape}, y={y_test.shape}")
            
            # Create datasets for binary classification
            train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
            valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
            test_dataset = TSDataset(X_test, y_test, scaler=True, scale_dim=[0])
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Use binary classifier instead of multi-label
            print(f"🔥 Starting binary fine-tuning for {epochs} epochs...")
            
            model_trainer = BinaryClassifier(model,
                                           train_loader, 
                                           valid_loader,
                                           learning_rate=lr,
                                           weight_decay=1e-5,
                                           name_scheduler='CosineAnnealingLR',
                                           n_epochs=epochs,
                                           early_stopping=True,
                                           patience=8,
                                           reduce_lr=True,
                                           factor=0.5,
                                           patience_reduce=3,
                                           verbose=True,
                                           device=self.device)
            
            # Train and evaluate
            train_losses, val_losses = model_trainer.fit()
            test_metrics = self.evaluate_binary_model(model, test_loader)
            
            # Convert single-appliance results to multi-label format for consistency
            multilabel_metrics = {successfully_loaded_cases[0]: {
                'accuracy': test_metrics['accuracy'],
                'f1_binary': test_metrics['f1_binary'], 
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'roc_auc': test_metrics['roc_auc']
            }}
            
            print(f"🎯 {successfully_loaded_cases[0]} Results:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
        else:
            print(f"✅ Successfully loaded {len(successfully_loaded_cases)} appliances: {successfully_loaded_cases}")
            
            # Update the model to match the actual number of successfully loaded appliances
            actual_nb_classes = len(successfully_loaded_cases)
            if actual_nb_classes != len(case_names):
                print(f"🔄 Updating model for {actual_nb_classes} classes instead of {len(case_names)}")
                
                # Recreate model with correct number of classes
                model = get_transapp_tst_model(m=1, 
                                              win=self.win, 
                                              dim_model=self.dim_model,
                                              mode="classif",
                                              nb_class=actual_nb_classes,  # Use actual number
                                              use_tst_pos_encoding=True,
                                              res_attention=True)
                
                # Reload pretrained weights with updated model
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                
                # Filter out classification head layers that have size mismatches
                filtered_dict = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            filtered_dict[k] = v
                        else:
                            print(f"⚠️ Skipping layer {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
                
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
                model = model.to(self.device)
                print(f"✅ Reloaded {len(filtered_dict)} compatible layers for {actual_nb_classes} classes")
                    
            X_train, X_valid, X_test = all_data
            y_train, y_valid, y_test = all_labels
            
            print(f"📊 Multi-label data shape:")
            print(f"  Train: X={X_train.shape}, y={y_train.shape}")
            print(f"  Valid: X={X_valid.shape}, y={y_valid.shape}")
            print(f"  Test: X={X_test.shape}, y={y_test.shape}")
            
            # Create datasets and dataloaders
            train_dataset = MultilabelTSDataset(X_train, y_train, scaler=True, scale_dim=[0])
            valid_dataset = MultilabelTSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
            test_dataset = MultilabelTSDataset(X_test, y_test, scaler=True, scale_dim=[0])
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Fine-tuning parameters
            dict_params = {
                'lr': lr,
                'wd': 1e-5,
                'batch_size': batch_size,
                'n_epochs': epochs,
                'early_stopping': True,
                'patience': 8,
                'reduce_lr': True,
                'factor': 0.5,
                'patience_reduce': 3
            }
            
            # Run fine-tuning with BCE loss for multi-label
            print(f"🔥 Starting multi-label fine-tuning for {epochs} epochs...")
            
            model_trainer = MultilabelClassifier(model,
                                               train_loader, 
                                               valid_loader,
                                               learning_rate=dict_params['lr'],
                                               weight_decay=dict_params['wd'],
                                               name_scheduler='CosineAnnealingLR',
                                               n_epochs=dict_params['n_epochs'],
                                               early_stopping=dict_params['early_stopping'],
                                               patience=dict_params['patience'],
                                               reduce_lr=dict_params['reduce_lr'],
                                               factor=dict_params['factor'],
                                               patience_reduce=dict_params['patience_reduce'],
                                               verbose=True,
                                               device=self.device)
            
            # Train and evaluate
            train_losses, val_losses = model_trainer.fit()
            multilabel_metrics = self.evaluate_multilabel_model(model, test_loader, successfully_loaded_cases)
        
        # Save the fine-tuned model
        save_path = self.results_dir / f"{self.experiment_name}_stage2_comstock_finetuned.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': multilabel_metrics if len(successfully_loaded_cases) > 1 else test_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                'win': self.win,
                'dim_model': self.dim_model,
                'nb_classes': actual_nb_classes,
                'appliances': successfully_loaded_cases,
                'stage': 'stage2_comstock_finetuning'
            }
        }, save_path)
        
        print(f"💾 Saved fine-tuned model to: {save_path}")
        
        # Store Stage 2 results 
        self.stage_2_results = {
            'metrics': multilabel_metrics if len(successfully_loaded_cases) > 1 else test_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'nb_appliances': len(successfully_loaded_cases),
            'appliances': successfully_loaded_cases,
            'model_path': str(save_path)
        }
        
        print(f"✅ Stage 2 completed successfully!")
        return model, save_path
    
    def stage3_comstock_evaluation(self, pretrained_model_path,
                                  case_names=None,
                                  resolution='30min',
                                  epochs=15, lr=1e-5, batch_size=32):
        """
        Stage 3: Final fine-tuning and evaluation on Comstock for all appliance cases
        """
        print("\n" + "="*60)
        print("🎯 STAGE 3: Comstock Final Fine-tuning & Evaluation")
        print("="*60)
        
        # Default to all available Comstock cases if not specified
        if case_names is None:
            case_names = [
                'cooling_case', 'exterior_lighting_case', 'fans_case',
                'heat_recovery_case', 'heat_rejection_case', 'heating_case',
                'interior_equipment_case', 'interior_lighting_case',
                'pumps_case', 'refrigeration_case', 'water_systems_case'
            ]
        
        # Load pretrained model
        print(f"📂 Loading pretrained model from: {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
        
        # ComStock has no exogenous variables (m=1)
        m = 1
        print("📊 Using only time series data (no exogenous variables for Comstock)")
        
        # Results for all cases
        all_case_results = {}
        
        # Process each appliance case
        for case_name in case_names:
            print(f"\n🔧 Processing {case_name}...")
            
            try:
                # Load Comstock data for this case
                data_tuple = COMSTOCK_get_data_case(
                    case_name=case_name,
                    resolution=resolution,
                    seed=0,
                    win=self.win,
                    ratio_resample=0.8
                )
                
                # Unpack the 12 returned values (including voter data)
                (X_train, y_train, X_valid, y_valid, X_test, y_test, 
                 X_train_voter, y_train_voter, X_valid_voter, y_valid_voter, 
                 X_test_voter, y_test_voter) = data_tuple
                
                # Check if we have sufficient data
                if len(X_train) == 0 or len(X_test) == 0:
                    print(f"⚠️ Skipping {case_name}: Insufficient training or test data")
                    all_case_results[case_name] = {
                        'error': 'Insufficient data',
                        'status': 'skipped',
                        'reason': 'No training or test samples available'
                    }
                    continue
                
                # Check if we have any positive samples
                positive_samples = y_train.sum()
                if positive_samples == 0:
                    print(f"⚠️ Skipping {case_name}: No positive samples in training data")
                    all_case_results[case_name] = {
                        'error': 'No positive samples',
                        'status': 'skipped',
                        'reason': 'No positive samples in training data'
                    }
                    continue
                
                print(f"📊 {case_name} data shape:")
                print(f"  Train: X={X_train.shape}, y={y_train.shape}")
                print(f"  Valid: X={X_valid.shape}, y={y_valid.shape}")
                print(f"  Test: X={X_test.shape}, y={y_test.shape}")
                print(f"  Positive ratio: {y_train.sum()/len(y_train):.3f}")
                
                # Create model for this case
                model = get_transapp_tst_model(m=m, 
                                              win=self.win, 
                                              dim_model=self.dim_model,
                                              mode="classif",
                                              nb_class=2,  # Binary classification
                                              use_tst_pos_encoding=False,  # No temporal encoding for Comstock
                                              res_attention=True)
                
                # Load pretrained weights
                if 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                else:
                    pretrained_dict = checkpoint
                    
                model_dict = model.state_dict()
                
                # Filter compatible layers
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                model = model.to(self.device)
                
                print(f"✅ Loaded {len(pretrained_dict)} compatible layers for {case_name}")
                
                # Create datasets and dataloaders
                train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
                valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
                test_dataset = TSDataset(X_test, y_test, scaler=True, scale_dim=[0])
                
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Fine-tuning parameters
                dict_params = {
                    'lr': lr,
                    'wd': 1e-5,
                    'batch_size': batch_size,
                    'n_epochs': epochs,
                    'early_stopping': True,
                    'patience': 10,
                    'reduce_lr': True,
                    'factor': 0.5,
                    'patience_reduce': 5
                }
                
                # Run fine-tuning for this case
                print(f"🔥 Starting {case_name} fine-tuning for {epochs} epochs...")
                
                model_trainer = BinaryClassifier(model,
                                               train_loader, 
                                               valid_loader,
                                               learning_rate=dict_params['lr'],
                                               weight_decay=dict_params['wd'],
                                               name_scheduler='CosineAnnealingLR',
                                               n_epochs=dict_params['n_epochs'],
                                               early_stopping=dict_params['early_stopping'],
                                               patience=dict_params['patience'],
                                               reduce_lr=dict_params['reduce_lr'],
                                               factor=dict_params['factor'],
                                               patience_reduce=dict_params['patience_reduce'],
                                               verbose=False,  # Reduce verbosity for multiple cases
                                               device=self.device)
                
                # Train and evaluate
                train_losses, val_losses = model_trainer.fit()
                test_metrics = self.evaluate_binary_model(model, test_loader)
                
                # Save model for this case
                case_save_path = self.results_dir / f"{self.experiment_name}_{case_name}_final.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'test_metrics': test_metrics,
                    'hyperparameters': dict_params,
                    'model_config': {
                        'm': m,
                        'win': self.win,
                        'dim_model': self.dim_model,
                        'case_name': case_name,
                        'resolution': resolution
                    }
                }, case_save_path)
                
                # Store results for this case
                all_case_results[case_name] = {
                    'dataset': f'Comstock_{resolution}',
                    'task': 'binary_classification',
                    'epochs': epochs,
                    'final_train_loss': train_losses[-1] if train_losses else None,
                    'final_val_loss': val_losses[-1] if val_losses else None,
                    'test_metrics': test_metrics,
                    'model_path': str(case_save_path),
                    'data_info': {
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'positive_ratio': float(y_train.sum()/len(y_train))
                    }
                }
                
                print(f"✅ Completed {case_name}: F1-Macro={test_metrics['f1_macro']:.4f}, F1-Binary={test_metrics['f1_binary']:.4f}")
                
            except Exception as e:
                print(f"❌ Failed to process {case_name}: {e}")
                all_case_results[case_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                continue
        
        # Store comprehensive results
        self.stage_results['stage3_comstock_evaluation'] = {
            'dataset': f'Comstock_{resolution}',
            'task': 'multi_case_binary_classification',
            'processed_cases': list(all_case_results.keys()),
            'successful_cases': [k for k, v in all_case_results.items() if 'error' not in v],
            'failed_cases': [k for k, v in all_case_results.items() if 'error' in v],
            'case_results': all_case_results
        }
        
        return all_case_results

    def stage3_cer_finetuning(self, pretrained_model_path, 
                             case_name='cooker_case', 
                             use_exogenous=True,
                             epochs=15, lr=1e-5, batch_size=32):
        """
        Stage 3: Final fine-tuning on CER for appliance ownership classification
        """
        print("\n" + "="*60)
        print("🎯 STAGE 3: CER Final Fine-tuning")
        print("="*60)
        
        # Load pretrained model
        print(f"📂 Loading pretrained model from: {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
        
        # Determine input dimension based on exogenous variable usage
        if use_exogenous:
            exo_variable = ['calendar', 'meteo_temperature']
            m = 3  # time series + 2 exogenous variables
            print("🌡️ Using exogenous variables: calendar + temperature")
        else:
            exo_variable = []
            m = 1  # only time series
            print("📊 Using only time series data (no exogenous variables)")
        
        # Create model for CER classification
        model = get_transapp_tst_model(m=m, 
                                      win=self.win, 
                                      dim_model=self.dim_model,
                                      mode="classif",
                                      nb_class=2,  # Binary classification
                                      use_tst_pos_encoding=True if use_exogenous else False,
                                      res_attention=True)
        
        # Load pretrained weights with dimension adaptation if needed
        # Handle different checkpoint structures (model_config vs config)
        if 'model_config' in checkpoint:
            checkpoint_m = checkpoint['model_config'].get('m', 1)
        elif 'config' in checkpoint:
            checkpoint_m = checkpoint['config'].get('m', 1)
        else:
            checkpoint_m = 1  # Default fallback
            
        if m != checkpoint_m:
            print(f"⚠️ Adapting model from m={checkpoint_m} to m={m}")
            # Load only compatible layers
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            
            # Filter out incompatible layers
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"✅ Loaded {len(pretrained_dict)} compatible layers")
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ Loaded all pretrained weights")
            
        model = model.to(self.device)
        
        # Load CER data
        print(f"📂 Loading CER data for {case_name}...")
        X_train, y_train, X_valid, y_valid, X_test, y_test = CER_get_data_case(
            case_name=case_name,
            seed=0,
            exo_variable=exo_variable,
            win=self.win,
            ratio_resample=0.8
        )
        
        print(f"📊 CER data shape:")
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Valid: X={X_valid.shape}, y={y_valid.shape}")
        print(f"  Test: X={X_test.shape}, y={y_test.shape}")
        print(f"  Positive ratio: {y_train.sum()/len(y_train):.3f}")
        
        # Create datasets and dataloaders
        train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
        valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
        test_dataset = TSDataset(X_test, y_test, scaler=True, scale_dim=[0])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Fine-tuning parameters
        dict_params = {
            'lr': lr,
            'wd': 1e-5,
            'batch_size': batch_size,
            'n_epochs': epochs,
            'early_stopping': True,
            'patience': 10,
            'reduce_lr': True,
            'factor': 0.5,
            'patience_reduce': 5
        }
        
        # Run final fine-tuning using AD_Framework (same as original TransApp)
        print(f"🔥 Starting CER fine-tuning for {epochs} epochs...")
        save_path = self.results_dir / f"{self.experiment_name}_stage3_cer_final.pt"
        
        # Use AD_Framework for consistency with original TransApp methodology
        model_trainer = AD_Framework(model,
                                   train_loader=train_loader, 
                                   valid_loader=valid_loader,
                                   learning_rate=dict_params['lr'],
                                   weight_decay=dict_params['wd'],
                                   criterion=nn.CrossEntropyLoss(),
                                   patience_es=dict_params['patience'], 
                                   patience_rlr=dict_params['patience_reduce'],
                                   f_metrics=getmetrics(),
                                   n_warmup_epochs=0,
                                   scale_by_subseq_in_voter=True, 
                                   scale_dim=[0],
                                   verbose=True, 
                                   plotloss=False,
                                   save_fig=False, 
                                   path_fig=None,
                                   device=self.device, 
                                   all_gpu=False,
                                   save_checkpoint=True, 
                                   path_checkpoint=save_path)
        
        # Train the model (original TransApp methodology)
        print(f"🏋️ Training CER model for {epochs} epochs...")
        model_trainer.train(dict_params['n_epochs'])
        
        # Evaluate on subsequences (original methodology)
        print("📊 Evaluating on test subsequences...")
        model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1), mask='test_metrics_lastmodel')
        
        # Restore best weights and get subsequence metrics
        model_trainer.restore_best_weights()
        subseq_metrics = model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))
        
        # Extract test metrics from subsequence evaluation
        test_metrics = {
            'accuracy': subseq_metrics.get('accuracy', 0.0),
            'f1_macro': subseq_metrics.get('f1_macro', 0.0),
            'f1_binary': subseq_metrics.get('f1_binary', 0.0),
            'precision': subseq_metrics.get('precision', 0.0), 
            'recall': subseq_metrics.get('recall', 0.0),
            'roc_auc': subseq_metrics.get('roc_auc', 0.0)
        }
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_metrics': test_metrics,
            'subseq_metrics': subseq_metrics,
            'hyperparameters': dict_params,
            'model_config': {
                'm': m,
                'win': self.win,
                'dim_model': self.dim_model,
                'case_name': case_name,
                'use_exogenous': use_exogenous,
                'exo_variables': exo_variable
            }
        }, save_path)
        
        print(f"💾 Saved final model to: {save_path}")
        
        # Store results
        self.stage_results['stage3_cer_finetuning'] = {
            'dataset': 'CER',
            'task': 'binary_classification',
            'case_name': case_name,
            'use_exogenous': use_exogenous,
            'epochs': epochs,
            'test_metrics': test_metrics,
            'subseq_metrics': subseq_metrics,
            'model_path': str(save_path)
        }
        
        return model, test_metrics
    
    def evaluate_binary_model(self, model, test_loader):
        """Evaluate binary classification model"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Get positive class probability
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        binary_preds = (all_preds > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, binary_preds),
            'f1_macro': f1_score(all_labels, binary_preds, average='macro'),
            'f1_binary': f1_score(all_labels, binary_preds),
            'precision': precision_score(all_labels, binary_preds),
            'recall': recall_score(all_labels, binary_preds),
            'roc_auc': roc_auc_score(all_labels, all_preds)
        }
        
        print(f"🎯 Test Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        return metrics
    
    def evaluate_multilabel_model(self, model, test_loader, case_names):
        """Evaluate multi-label classification model"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_x)
                probs = torch.sigmoid(outputs)  # Multi-label uses sigmoid
                
                all_preds.append(probs.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate metrics per appliance
        metrics = {}
        for i, case_name in enumerate(case_names):
            preds = (all_preds[:, i] > 0.5).astype(int)
            labels = all_labels[:, i]
            
            case_metrics = {
                'accuracy': accuracy_score(labels, preds),
                'f1_binary': f1_score(labels, preds),
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'roc_auc': roc_auc_score(labels, all_preds[:, i]) if len(np.unique(labels)) > 1 else 0
            }
            
            metrics[case_name] = case_metrics
            print(f"🎯 {case_name} Results:")
            for metric, value in case_metrics.items():
                print(f"  {metric}: {value:.4f}")
                
        return metrics
    
    def run_full_experiment(self, skip_stage1=False, skip_stage2=False,
                           pretrained_path=None, stage2_path=None,
                           comstock_evaluation_mode=False):
        """
        Run the complete multi-stage experiment
        """
        print("\n" + "="*80)
        if comstock_evaluation_mode:
            print("🚀 STARTING COMSTOCK EVALUATION EXPERIMENT")
        else:
            print("🚀 STARTING MULTI-STAGE TST EXPERIMENT")
        print("="*80)
        
        start_time = datetime.now()
        
        # For Comstock evaluation mode, skip stages 1 and 2, go directly to Comstock evaluation
        if comstock_evaluation_mode:
            if not pretrained_path:
                raise ValueError("pretrained_path must be provided for Comstock evaluation mode")
            
            print(f"🎯 Comstock Evaluation Mode: Using pretrained model from {pretrained_path}")
            
            # Run Comstock evaluation on all cases
            stage3_results = self.stage3_comstock_evaluation(
                pretrained_model_path=pretrained_path
            )
            
            # Calculate summary metrics
            successful_cases = [k for k, v in stage3_results.items() if 'error' not in v]
            failed_cases = [k for k, v in stage3_results.items() if 'error' in v]
            
            if successful_cases:
                avg_f1_macro = np.mean([stage3_results[case]['test_metrics']['f1_macro'] 
                                       for case in successful_cases])
                avg_f1_binary = np.mean([stage3_results[case]['test_metrics']['f1_binary'] 
                                        for case in successful_cases])
                avg_roc_auc = np.mean([stage3_results[case]['test_metrics']['roc_auc'] 
                                      for case in successful_cases])
                
                summary_metrics = {
                    'average_f1_macro': avg_f1_macro,
                    'average_f1_binary': avg_f1_binary,
                    'average_roc_auc': avg_roc_auc,
                    'successful_cases': len(successful_cases),
                    'failed_cases': len(failed_cases),
                    'total_cases': len(stage3_results)
                }
            else:
                summary_metrics = {
                    'error': 'No successful cases',
                    'failed_cases': len(failed_cases),
                    'total_cases': len(stage3_results)
                }
            
        else:
            # Original multi-stage experiment logic
            # Stage 1: ComStock pretraining
            if not skip_stage1:
                stage1_model, stage1_path = self.stage1_comstock_pretraining(
                    resolution='30min',
                    epochs=15,
                    lr=1e-4,
                    batch_size=64,
                    mask_ratio=0.4
                )
            else:
                stage1_path = pretrained_path
                print(f"⏭️ Skipping Stage 1, using provided model: {stage1_path}")
            
            # Stage 2: ComStock fine-tuning (optional)
            if not skip_stage2:
                stage2_model, stage2_path = self.stage2_comstock_finetuning(
                    pretrained_model_path=stage1_path,
                    resolution='30min',
                    case_names=['cooling_case', 'fans_case', 'pumps_case'],
                    epochs=15,
                    lr=1e-5,
                    batch_size=32
                )
            else:
                stage2_path = stage2_path or stage1_path
                print(f"⏭️ Skipping Stage 2, using model: {stage2_path}")
            
            # Stage 3: CER final fine-tuning
            stage3_model, stage3_metrics = self.stage3_cer_finetuning(
                pretrained_model_path=stage2_path,
                case_name='cooker_case',
                use_exogenous=True,
                epochs=15,
                lr=1e-5,
                batch_size=32
            )
            summary_metrics = stage3_metrics
        
        # Calculate total time
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # Save comprehensive results
        final_results = {
            'experiment_name': self.experiment_name,
            'timestamp': start_time.isoformat(),
            'total_duration': str(total_time),
            'mode': 'comstock_evaluation' if comstock_evaluation_mode else 'multi_stage',
            'configuration': {
                'win': self.win,
                'dim_model': self.dim_model,
                'device': str(self.device)
            },
            'stage_results': self.stage_results,
            'summary_metrics': summary_metrics
        }
        
        results_file = self.results_dir / f"{self.experiment_name}_comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print("\n" + "="*80)
        if comstock_evaluation_mode:
            print("🎉 COMSTOCK EVALUATION COMPLETE!")
            if 'average_f1_macro' in summary_metrics:
                print(f"📊 Average F1-Macro across all cases: {summary_metrics['average_f1_macro']:.4f}")
                print(f"🎯 Average F1-Binary across all cases: {summary_metrics['average_f1_binary']:.4f}")
                print(f"📈 Average ROC-AUC across all cases: {summary_metrics['average_roc_auc']:.4f}")
                print(f"✅ Successful cases: {summary_metrics['successful_cases']}/{summary_metrics['total_cases']}")
        else:
            print("🎉 MULTI-STAGE EXPERIMENT COMPLETE!")
            if isinstance(summary_metrics, dict) and 'f1_macro' in summary_metrics:
                print(f"🏆 Final F1-Macro: {summary_metrics['f1_macro']:.4f}")
                print(f"🎯 Final F1-Binary: {summary_metrics['f1_binary']:.4f}")
                print(f"📈 Final ROC-AUC: {summary_metrics['roc_auc']:.4f}")
        print("="*80)
        print(f"⏱️ Total time: {total_time}")
        print(f"💾 Results saved to: {results_file}")
        
        return final_results


# ============================================================================
# CUSTOM TRAINING CLASSES FOR MULTI-STAGE EXPERIMENTS
# ============================================================================

class MAETSDataset(torch.utils.data.Dataset):
    """Dataset for Masked Autoencoder pretraining"""
    
    def __init__(self, X, mask_ratio=0.4, scaler=True, scale_dim=[0]):
        self.X = X
        self.mask_ratio = mask_ratio
        
        if scaler:
            self.scaler = StandardScaler()
            # Fit scaler on specified dimensions
            for dim in scale_dim:
                X_dim = X[:, dim, :].reshape(-1, X.shape[-1])
                self.scaler.fit(X_dim)
                X[:, dim, :] = self.scaler.transform(X_dim).reshape(X.shape[0], -1)
        
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        # Create random mask
        seq_len = x.shape[-1]
        mask_len = int(seq_len * self.mask_ratio)
        mask_indices = torch.randperm(seq_len)[:mask_len]
        
        # Create masked input
        x_masked = x.clone()
        x_masked[:, mask_indices] = 0  # Mask to zero
        
        return x_masked, x  # Return masked input and original as target


class MultilabelTSDataset(torch.utils.data.Dataset):
    """Dataset for multi-label time series classification"""
    
    def __init__(self, X, y, scaler=True, scale_dim=[0]):
        if scaler:
            scaler_obj = StandardScaler()
            for dim in scale_dim:
                X_dim = X[:, dim, :].reshape(-1, X.shape[-1])
                scaler_obj.fit(X_dim)
                X[:, dim, :] = scaler_obj.transform(X_dim).reshape(X.shape[0], -1)
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MAEPretrainer:
    """Masked Autoencoder pretrainer for TST models"""
    
    def __init__(self, model, train_loader, valid_loader=None, learning_rate=1e-4,
                 weight_decay=1e-5, name_scheduler='CosineAnnealingLR', n_epochs=50,
                 early_stopping=True, patience=10, reduce_lr=True, factor=0.5,
                 patience_reduce=5, verbose=True, device='cuda'):
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.verbose = verbose
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), 
                                          lr=learning_rate, 
                                          weight_decay=weight_decay)
        
        if name_scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs
            )
        
        # Training parameters
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.reduce_lr = reduce_lr
        self.factor = factor
        self.patience_reduce = patience_reduce
        
        # Loss function - MSE for reconstruction
        self.criterion = nn.MSELoss()
        
    def fit(self):
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        patience_count = 0
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_target in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_target = batch_target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            # Validation (if available)
            val_loss = 0.0
            if self.valid_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch_x, batch_target in self.valid_loader:
                        batch_x = batch_x.to(self.device)
                        batch_target = batch_target.to(self.device)
                        
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_target)
                        val_loss += loss.item()
                
                val_loss /= len(self.valid_loader)
                val_losses.append(val_loss)
                current_loss = val_loss
            else:
                current_loss = train_loss
            
            # Scheduler step
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_count = 0
            else:
                patience_count += 1
                
            if self.verbose and epoch % 10 == 0:
                if self.valid_loader:
                    print(f"Epoch {epoch:3d}/{self.n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch:3d}/{self.n_epochs} - Train Loss: {train_loss:.6f}")
                    
            if self.early_stopping and patience_count >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return train_losses, val_losses


class MultilabelClassifier:
    """Multi-label classifier trainer for ComStock fine-tuning"""
    
    def __init__(self, model, train_loader, valid_loader, learning_rate=1e-5,
                 weight_decay=1e-5, name_scheduler='CosineAnnealingLR', n_epochs=30,
                 early_stopping=True, patience=8, reduce_lr=True, factor=0.5,
                 patience_reduce=3, verbose=True, device='cuda'):
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.verbose = verbose
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), 
                                          lr=learning_rate, 
                                          weight_decay=weight_decay)
        
        if name_scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs
            )
        
        # Training parameters
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        
        # Loss function - BCE for multi-label
        self.criterion = nn.BCEWithLogitsLoss()
        
    def fit(self):
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        patience_count = 0
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in self.valid_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(self.valid_loader)
            val_losses.append(val_loss)
            
            # Scheduler step
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
            else:
                patience_count += 1
                
            if self.verbose and epoch % 5 == 0:
                print(f"Epoch {epoch:3d}/{self.n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
            if self.early_stopping and patience_count >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return train_losses, val_losses


class BinaryClassifier:
    """Binary classifier trainer for final CER fine-tuning"""
    
    def __init__(self, model, train_loader, valid_loader, learning_rate=1e-5,
                 weight_decay=1e-5, name_scheduler='CosineAnnealingLR', n_epochs=15,
                 early_stopping=True, patience=10, reduce_lr=True, factor=0.5,
                 patience_reduce=5, verbose=True, device='cuda'):
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.verbose = verbose
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), 
                                          lr=learning_rate, 
                                          weight_decay=weight_decay)
        
        if name_scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs
            )
        
        # Training parameters
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        
        # Loss function - CrossEntropy for binary classification
        self.criterion = nn.CrossEntropyLoss()
        
    def fit(self):
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        patience_count = 0
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).long()
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in self.valid_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device).long()
                    
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(self.valid_loader)
            val_losses.append(val_loss)
            
            # Scheduler step
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
            else:
                patience_count += 1
                
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{self.n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
            if self.early_stopping and patience_count >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return train_losses, val_losses


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Stage TST Experiments")
    
    parser.add_argument('--experiment_name', type=str, default='multistage_tst',
                       help='Name of the experiment')
    parser.add_argument('--win', type=int, default=1024,
                       help='Window size for time series')
    parser.add_argument('--dim_model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    # Stage control
    parser.add_argument('--skip_stage1', action='store_true',
                       help='Skip ComStock pretraining stage')
    parser.add_argument('--skip_stage2', action='store_true',
                       help='Skip ComStock fine-tuning stage')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model (if skipping stage 1)')
    parser.add_argument('--stage2_path', type=str, default=None,
                       help='Path to stage 2 model (if skipping stage 2)')
    
    # Comstock evaluation mode
    parser.add_argument('--comstock_evaluation', action='store_true',
                       help='Run Comstock evaluation mode (all appliance cases)')
    
    # CER fine-tuning options
    parser.add_argument('--cer_case', type=str, default='cooker_case',
                       help='CER case name for final fine-tuning')
    parser.add_argument('--no_exogenous', action='store_true',
                       help='Disable exogenous variables for CER')
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = MultiStageTSTExperiment(
        experiment_name=args.experiment_name,
        win=args.win,
        dim_model=args.dim_model,
        device=args.device
    )
    
    # Run experiment based on mode
    if args.comstock_evaluation:
        if not args.pretrained_path:
            print("❌ Error: --pretrained_path is required for Comstock evaluation mode")
            return None
            
        results = experiment.run_full_experiment(
            comstock_evaluation_mode=True,
            pretrained_path=args.pretrained_path
        )
    else:
        # Run full experiment
        results = experiment.run_full_experiment(
            skip_stage1=args.skip_stage1,
            skip_stage2=args.skip_stage2,
            pretrained_path=args.pretrained_path,
            stage2_path=args.stage2_path
        )
    
    def evaluate_binary_model(self, model, test_loader):
        """Evaluate a binary classification model"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).long()
                
                outputs = model(X_batch)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_binary = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_probs)
        
        return {
            'accuracy': accuracy,
            'f1_binary': f1_binary,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

    print("\n🎉 Multi-Stage TST Experiment Complete!")
    return results


if __name__ == "__main__":
    main()