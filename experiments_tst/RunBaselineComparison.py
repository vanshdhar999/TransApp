#################################################################################################################
#
# @copyright : ©2023 EDF  
# @author : Assistant (Created for comprehensive baseline comparison)
# @description : Comprehensive baseline comparison for Multi-Stage TST experiments
# @component: experiments_tst/
# @file : RunBaselineComparison.py
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
    from src.TransAppModel.TransApp import TransApp
    from src.TransAppModel.TransApp_TST import *
    from src.AD_Framework.Framework import *
    from src.utils.losses import *
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Root directory: {root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

class BaselineComparison:
    """
    Comprehensive baseline comparison for Multi-Stage TST experiments
    
    Compares:
    1. Standard TransApp (original architecture)
    2. TST from scratch (no pretraining)
    3. CER-only pretraining + fine-tuning
    4. ComStock-only pretraining + evaluation
    5. Multi-stage TST (our approach)
    """
    
    def __init__(self, experiment_name="baseline_comparison", 
                 win=1024, dim_model=96, device='auto'):
        self.experiment_name = experiment_name
        self.win = win
        self.dim_model = dim_model
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🔬 Baseline Comparison Experiment: {experiment_name}")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Window size: {win}, Model dimension: {dim_model}")
        
        # Create results directory
        self.results_dir = Path(root) / "results" / "BaselineComparison"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Track all baseline results
        self.baseline_results = {}
        
    def baseline1_standard_transapp(self, case_name='cooker_case', use_exogenous=True,
                                   epochs=15, lr=1e-4, batch_size=32):
        """
        Baseline 1: Standard TransApp (original architecture)
        """
        print("\n" + "="*60)
        print("🔵 BASELINE 1: Standard TransApp")
        print("="*60)
        
        # Determine input dimension based on exogenous variable usage
        if use_exogenous:
            exo_variable = ['calendar', 'meteo_temperature']
            m = 3  # time series + 2 exogenous variables
            print("🌡️ Using exogenous variables: calendar + temperature")
        else:
            exo_variable = []
            m = 1  # only time series
            print("📊 Using only time series data (no exogenous variables)")
        
        # Create standard TransApp model
        model = TransApp(
            max_len=self.win, c_in=m, mode="classif",
            n_embed_blocks=1, encoding_type="noencoding",
            n_encoder_layers=3, kernel_size=5, d_model=self.dim_model,
            pffn_ratio=2, n_head=4, prenorm=True, norm="LayerNorm",
            activation='gelu', store_att=False, attn_dp_rate=0.2,
            head_dp_rate=0.1, dp_rate=0.2,
            att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 
                      'learnable_scale_enc': False},
            c_reconstruct=1, apply_gap=True, nb_class=2
        ).to(self.device)
        
        print(f"🏗️ Created Standard TransApp model:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
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
        
        # Training parameters
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
        
        # Train standard TransApp
        print(f"🔥 Training Standard TransApp for {epochs} epochs...")
        save_path = self.results_dir / f"baseline1_standard_transapp_{case_name}.pt"
        
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
                                       verbose=True,
                                       device=self.device)
        
        # Train and evaluate
        train_losses, val_losses = model_trainer.fit()
        test_metrics = self.evaluate_binary_model(model, test_loader)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_metrics': test_metrics,
            'hyperparameters': dict_params,
            'model_config': {
                'architecture': 'Standard_TransApp',
                'm': m,
                'win': self.win,
                'dim_model': self.dim_model,
                'case_name': case_name,
                'use_exogenous': use_exogenous,
                'exo_variables': exo_variable
            }
        }, save_path)
        
        # Store results
        self.baseline_results['baseline1_standard_transapp'] = {
            'architecture': 'Standard_TransApp',
            'dataset': 'CER',
            'task': 'binary_classification',
            'case_name': case_name,
            'use_exogenous': use_exogenous,
            'epochs': epochs,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'test_metrics': test_metrics,
            'model_path': str(save_path),
            'parameter_count': sum(p.numel() for p in model.parameters())
        }
        
        print(f"✅ Baseline 1 Complete: F1-Macro={test_metrics['f1_macro']:.4f}, F1-Binary={test_metrics['f1_binary']:.4f}")
        return test_metrics
        
    def baseline2_tst_from_scratch(self, case_name='cooker_case', use_exogenous=True,
                                  epochs=15, lr=1e-4, batch_size=32):
        """
        Baseline 2: TST from scratch (no pretraining)
        """
        print("\n" + "="*60)
        print("🟡 BASELINE 2: TST From Scratch")
        print("="*60)
        
        # Determine input dimension based on exogenous variable usage
        if use_exogenous:
            exo_variable = ['calendar', 'meteo_temperature']
            m = 3  # time series + 2 exogenous variables
            print("🌡️ Using exogenous variables: calendar + temperature")
        else:
            exo_variable = []
            m = 1  # only time series
            print("📊 Using only time series data (no exogenous variables)")
        
        # Create TST model without pretraining
        model = get_transapp_tst_model(m=m, 
                                      win=self.win, 
                                      dim_model=self.dim_model,
                                      mode="classif",
                                      nb_class=2,
                                      use_tst_pos_encoding=True if use_exogenous else False,
                                      norm="BatchNorm",
                                      res_attention=True).to(self.device)
        
        print(f"🏗️ Created TST model (from scratch):")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load CER data
        print(f"📂 Loading CER data for {case_name}...")
        X_train, y_train, X_valid, y_valid, X_test, y_test = CER_get_data_case(
            case_name=case_name,
            seed=0,
            exo_variable=exo_variable,
            win=self.win,
            ratio_resample=0.8
        )
        
        # Create datasets and dataloaders
        train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
        valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
        test_dataset = TSDataset(X_test, y_test, scaler=True, scale_dim=[0])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Training parameters
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
        
        # Train TST from scratch
        print(f"🔥 Training TST from scratch for {epochs} epochs...")
        save_path = self.results_dir / f"baseline2_tst_scratch_{case_name}.pt"
        
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
                                       verbose=True,
                                       device=self.device)
        
        # Train and evaluate
        train_losses, val_losses = model_trainer.fit()
        test_metrics = self.evaluate_binary_model(model, test_loader)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_metrics': test_metrics,
            'hyperparameters': dict_params,
            'model_config': {
                'architecture': 'TST_TransApp',
                'm': m,
                'win': self.win,
                'dim_model': self.dim_model,
                'case_name': case_name,
                'use_exogenous': use_exogenous,
                'pretraining': 'none',
                'exo_variables': exo_variable
            }
        }, save_path)
        
        # Store results
        self.baseline_results['baseline2_tst_scratch'] = {
            'architecture': 'TST_TransApp',
            'dataset': 'CER',
            'task': 'binary_classification',
            'case_name': case_name,
            'use_exogenous': use_exogenous,
            'pretraining': 'none',
            'epochs': epochs,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'test_metrics': test_metrics,
            'model_path': str(save_path),
            'parameter_count': sum(p.numel() for p in model.parameters())
        }
        
        print(f"✅ Baseline 2 Complete: F1-Macro={test_metrics['f1_macro']:.4f}, F1-Binary={test_metrics['f1_binary']:.4f}")
        return test_metrics
        
    def baseline3_cer_only_pretraining(self, case_name='cooker_case', use_exogenous=True,
                                      pretraining_epochs=20, finetuning_epochs=15, 
                                      lr=1e-4, batch_size=32):
        """
        Baseline 3: CER-only pretraining + fine-tuning
        """
        print("\n" + "="*60)
        print("🟢 BASELINE 3: CER-Only Pretraining")
        print("="*60)
        
        # Use existing CER-pretrained model
        if use_exogenous:
            pretrained_path = str(root) + '/results/TransAppPretrained_TST/Embed/TransApp_TST_96_BatchNorm.pt'
            exo_variable = ['calendar', 'meteo_temperature']
            m = 3
            print("🌡️ Using CER-pretrained model with exogenous variables")
        else:
            pretrained_path = str(root) + '/results/TransAppPretrained_TST/None/TransApp_TST_96_BatchNorm.pt'
            exo_variable = []
            m = 1
            print("📊 Using CER-pretrained model without exogenous variables")
        
        # Check if pretrained model exists
        if not os.path.exists(pretrained_path):
            print(f"⚠️ Pretrained model not found: {pretrained_path}")
            print("   This baseline requires running CER pretraining first")
            return None
            
        # Load pretrained model
        print(f"📂 Loading CER-pretrained model from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        # Create model for fine-tuning
        model = get_transapp_tst_model(m=m, 
                                      win=self.win, 
                                      dim_model=self.dim_model,
                                      mode="classif",
                                      nb_class=2,
                                      use_tst_pos_encoding=True if use_exogenous else False,
                                      norm="BatchNorm",
                                      res_attention=True)
        
        # Load pretrained weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        model = model.to(self.device)
        print(f"✅ Loaded CER-pretrained weights")
        
        # Load CER data for fine-tuning
        print(f"📂 Loading CER data for {case_name} fine-tuning...")
        X_train, y_train, X_valid, y_valid, X_test, y_test = CER_get_data_case(
            case_name=case_name,
            seed=0,
            exo_variable=exo_variable,
            win=self.win,
            ratio_resample=0.8
        )
        
        # Create datasets and dataloaders
        train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
        valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
        test_dataset = TSDataset(X_test, y_test, scaler=True, scale_dim=[0])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Fine-tuning parameters (lower learning rate)
        dict_params = {
            'lr': lr / 10,  # Lower LR for fine-tuning
            'wd': 1e-5,
            'batch_size': batch_size,
            'n_epochs': finetuning_epochs,
            'early_stopping': True,
            'patience': 10,
            'reduce_lr': True,
            'factor': 0.5,
            'patience_reduce': 5
        }
        
        # Fine-tune on CER
        print(f"🔥 Fine-tuning CER-pretrained model for {finetuning_epochs} epochs...")
        save_path = self.results_dir / f"baseline3_cer_only_{case_name}.pt"
        
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
                                       verbose=True,
                                       device=self.device)
        
        # Train and evaluate
        train_losses, val_losses = model_trainer.fit()
        test_metrics = self.evaluate_binary_model(model, test_loader)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_metrics': test_metrics,
            'hyperparameters': dict_params,
            'model_config': {
                'architecture': 'TST_TransApp',
                'm': m,
                'win': self.win,
                'dim_model': self.dim_model,
                'case_name': case_name,
                'use_exogenous': use_exogenous,
                'pretraining': 'CER_only',
                'pretrained_path': pretrained_path,
                'exo_variables': exo_variable
            }
        }, save_path)
        
        # Store results
        self.baseline_results['baseline3_cer_only'] = {
            'architecture': 'TST_TransApp',
            'dataset': 'CER',
            'task': 'binary_classification',
            'case_name': case_name,
            'use_exogenous': use_exogenous,
            'pretraining': 'CER_only',
            'epochs': finetuning_epochs,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'test_metrics': test_metrics,
            'model_path': str(save_path),
            'parameter_count': sum(p.numel() for p in model.parameters())
        }
        
        print(f"✅ Baseline 3 Complete: F1-Macro={test_metrics['f1_macro']:.4f}, F1-Binary={test_metrics['f1_binary']:.4f}")
        return test_metrics
        
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
        
    def run_comprehensive_baseline_comparison(self, case_name='cooker_case', 
                                            use_exogenous=True, epochs=15):
        """
        Run all baseline experiments for comprehensive comparison
        """
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE BASELINE COMPARISON")
        print("="*80)
        
        start_time = datetime.now()
        
        # Run all baselines
        print(f"Testing case: {case_name}")
        print(f"Using exogenous variables: {use_exogenous}")
        print(f"Training epochs: {epochs}")
        
        # Baseline 1: Standard TransApp
        try:
            self.baseline1_standard_transapp(case_name, use_exogenous, epochs)
        except Exception as e:
            print(f"❌ Baseline 1 failed: {e}")
            self.baseline_results['baseline1_standard_transapp'] = {'error': str(e)}
        
        # Baseline 2: TST from scratch
        try:
            self.baseline2_tst_from_scratch(case_name, use_exogenous, epochs)
        except Exception as e:
            print(f"❌ Baseline 2 failed: {e}")
            self.baseline_results['baseline2_tst_scratch'] = {'error': str(e)}
        
        # Baseline 3: CER-only pretraining
        try:
            self.baseline3_cer_only_pretraining(case_name, use_exogenous, 
                                               pretraining_epochs=20, 
                                               finetuning_epochs=epochs)
        except Exception as e:
            print(f"❌ Baseline 3 failed: {e}")
            self.baseline_results['baseline3_cer_only'] = {'error': str(e)}
        
        # Calculate total time
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # Save comprehensive results
        final_results = {
            'experiment_name': self.experiment_name,
            'timestamp': start_time.isoformat(),
            'total_duration': str(total_time),
            'configuration': {
                'case_name': case_name,
                'use_exogenous': use_exogenous,
                'win': self.win,
                'dim_model': self.dim_model,
                'epochs': epochs,
                'device': str(self.device)
            },
            'baseline_results': self.baseline_results
        }
        
        results_file = self.results_dir / f"{self.experiment_name}_comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Print comparison summary
        self.print_comparison_summary()
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"⏱️ Total time: {total_time}")
        
        return final_results
        
    def print_comparison_summary(self):
        """Print a summary table comparing all baselines"""
        print("\n" + "="*80)
        print("📊 BASELINE COMPARISON SUMMARY")
        print("="*80)
        
        # Extract successful results
        successful_baselines = {k: v for k, v in self.baseline_results.items() 
                              if 'error' not in v and 'test_metrics' in v}
        
        if not successful_baselines:
            print("❌ No successful baselines to compare")
            return
            
        print(f"{'Baseline':<25} {'Architecture':<15} {'F1-Macro':<10} {'F1-Binary':<10} {'ROC-AUC':<10} {'Params':<12}")
        print("-" * 90)
        
        for baseline_name, results in successful_baselines.items():
            name = baseline_name.replace('baseline', 'B').replace('_', ' ').title()
            arch = results.get('architecture', 'Unknown')
            metrics = results.get('test_metrics', {})
            params = f"{results.get('parameter_count', 0):,}"
            
            f1_macro = f"{metrics.get('f1_macro', 0):.4f}"
            f1_binary = f"{metrics.get('f1_binary', 0):.4f}"
            roc_auc = f"{metrics.get('roc_auc', 0):.4f}"
            
            print(f"{name:<25} {arch:<15} {f1_macro:<10} {f1_binary:<10} {roc_auc:<10} {params:<12}")
        
        # Find best performing baseline
        if successful_baselines:
            best_baseline = max(successful_baselines.items(), 
                              key=lambda x: x[1]['test_metrics'].get('f1_macro', 0))
            print(f"\n🏆 Best performing baseline: {best_baseline[0]}")
            print(f"   F1-Macro: {best_baseline[1]['test_metrics']['f1_macro']:.4f}")


# Import custom training classes from RunMultiStageTST.py
class BinaryClassifier:
    """Binary classifier trainer for baseline experiments"""
    
    def __init__(self, model, train_loader, valid_loader, learning_rate=1e-5,
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


def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description="Baseline Comparison for Multi-Stage TST")
    
    parser.add_argument('--experiment_name', type=str, default='baseline_comparison',
                       help='Name of the experiment')
    parser.add_argument('--case_name', type=str, default='cooker_case',
                       help='CER case name for comparison')
    parser.add_argument('--win', type=int, default=1024,
                       help='Window size for time series')
    parser.add_argument('--dim_model', type=int, default=96,
                       help='Model dimension')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--no_exogenous', action='store_true',
                       help='Disable exogenous variables')
    
    args = parser.parse_args()
    
    # Create baseline comparison
    comparison = BaselineComparison(
        experiment_name=args.experiment_name,
        win=args.win,
        dim_model=args.dim_model,
        device=args.device
    )
    
    # Run comprehensive comparison
    results = comparison.run_comprehensive_baseline_comparison(
        case_name=args.case_name,
        use_exogenous=not args.no_exogenous,
        epochs=args.epochs
    )
    
    print("\n🎉 Baseline Comparison Complete!")
    return results


if __name__ == "__main__":
    main()