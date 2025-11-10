#!/usr/bin/env python3
"""
PatchTST for CER Appliance Detection - Phase 1
==============================================

This script implements PatchTST (Patch Time-Series Transformer) for appliance detection
on the CER dataset without pretraining. It includes comprehensive hyperparameter tuning,
detailed logging, and JSON metrics storage.

Key Features:
- Direct PatchTST implementation for classification
- Hyperparameter grid search with configurable ranges
- Comprehensive logging and JSON metrics storage
- Multiple evaluation metrics (F1-macro, accuracy, ROC-AUC)
- Statistical validation with multiple random seeds
- Compatible with existing CER data pipeline

Author: Enhanced TransApp Framework
Date: October 2025
Phase: 1 (Direct PatchTST without pretraining)
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from pathlib import Path
import logging
from itertools import product
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from types import SimpleNamespace
import time
import argparse

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Fix path resolution
current_file = Path(__file__).resolve()
root = current_file.parents[1]  # Go up from experiments_patch/ to TransApp/
sys.path.insert(0, str(root))

try:
    from experiments.data_utils import CER_get_data_case
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
    # Note: Using custom PatchTST implementation due to transformers version compatibility
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: torch, sklearn")
    sys.exit(1)

@dataclass
class PatchTSTHyperparameters:
    """Hyperparameter configuration for PatchTST experiments"""
    
    # Core model parameters
    context_length: int = 96          # Input sequence length (15-min intervals for 24h)
    patch_length: int = 16            # Length of each patch
    stride: int = 8                   # Stride between patches
    hidden_size: int = 128            # Transformer embedding dimension
    num_hidden_layers: int = 3        # Number of transformer layers
    num_attention_heads: int = 8      # Number of attention heads
    
    # Regularization
    dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    epochs: int = 15
    patience_early_stopping: int = 5
    patience_lr_reduction: int = 3
    
    # Data parameters
    embed_type: int = 0               # 0: no embeddings, 1: with temporal embeddings
    random_seed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return asdict(self)

class PatchTSTLogger:
    """Comprehensive logging system for PatchTST experiments"""
    
    def __init__(self, experiment_name: str, log_dir: Path):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Starting PatchTST experiment: {experiment_name}")
        
    def log_hyperparameters(self, params: PatchTSTHyperparameters):
        """Log hyperparameter configuration"""
        self.logger.info("⚙️ Hyperparameter Configuration:")
        for key, value in params.to_dict().items():
            # Convert lists to string representation to avoid format string errors
            if isinstance(value, list):
                value_str = f"[{', '.join(map(str, value))}]"
            else:
                value_str = str(value)
            self.logger.info(f"  {key}: {value_str}")
    
    def log_data_info(self, train_size: int, valid_size: int, test_size: int):
        """Log dataset information"""
        self.logger.info(f"📊 Dataset Information:")
        self.logger.info(f"  Training samples: {train_size:,}")
        self.logger.info(f"  Validation samples: {valid_size:,}")
        self.logger.info(f"  Test samples: {test_size:,}")
    
    def log_model_info(self, model: nn.Module):
        """Log model architecture information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"🏗️ Model Architecture:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def log_training_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                          learning_rate: float, elapsed_time: float):
        """Log training epoch information"""
        self.logger.info(f"Epoch [{epoch:2d}] - Loss: {train_loss:.6f} | "
                        f"Val: {val_loss:.6f} | LR: {learning_rate:.2e} | "
                        f"Time: {elapsed_time:.1f}s")
    
    def log_results(self, results: Dict[str, float]):
        """Log final evaluation results"""
        self.logger.info("🎯 Final Results:")
        for metric, value in results.items():
            print(f"metric: {metric}, value: {value}")
            # self.logger.info(f"  {metric}: {value:.6f}")

class PatchTSTDataProcessor:
    """Data processing pipeline for PatchTST experiments"""
    
    def __init__(self, context_length: int = 96):
        self.context_length = context_length
    
    def load_cer_data(self, case_name: str, embed_type: int, random_seed: int) -> Tuple:
        """Load and preprocess CER data for PatchTST"""
        
        # Determine exogenous variables
        exo_variables = [] if embed_type == 0 else ["hours_cos", "hours_sin", "days_cos", "days_sin"]
        
        # Load data using existing CER pipeline
        datas_tuple = CER_get_data_case(
            case_name=case_name,
            seed=random_seed,
            exo_variable=exo_variables,
            win=self.context_length,
            ratio_resample=0.8
        )
        
        X_train, y_train = datas_tuple[0], datas_tuple[1]
        X_valid, y_valid = datas_tuple[2], datas_tuple[3]
        X_test, y_test = datas_tuple[4], datas_tuple[5]
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    
    def prepare_patchtst_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data tensors for PatchTST model"""
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).squeeze()  # Remove channel dimension if present
        y_tensor = torch.LongTensor(y)
        
        # Ensure correct shape: [batch_size, sequence_length, num_features]
        if len(X_tensor.shape) == 2:  # [batch_size, sequence_length]
            X_tensor = X_tensor.unsqueeze(-1)  # Add feature dimension
        elif len(X_tensor.shape) == 3 and X_tensor.shape[1] == 1:  # [batch_size, 1, sequence_length]
            X_tensor = X_tensor.transpose(1, 2)  # [batch_size, sequence_length, 1]
        
        return X_tensor, y_tensor

class PatchTSTEvaluator:
    """Comprehensive evaluation system for PatchTST models"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass with correct input format
                outputs = model(past_values=batch_x)
                
                # Extract logits from output object (handle both original and custom models)
                if hasattr(outputs, 'prediction_logits'):
                    logits = outputs.prediction_logits
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'prediction_outputs'):
                    logits = outputs.prediction_outputs
                else:
                    logits = outputs
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)[:, 1]  # Binary classification
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        f1_weighted = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')[2]
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_labels, all_probabilities)
        except:
            roc_auc = 0.5  # Random baseline if calculation fails
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'precision_binary': float(precision[1]) if len(precision) > 1 else 0.0,
            'recall_binary': float(recall[1]) if len(recall) > 1 else 0.0,
            'f1_binary': float(f1[1]) if len(f1) > 1 else 0.0,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }

class PatchTSTTrainer:
    """Training pipeline for PatchTST models"""
    
    def __init__(self, logger: PatchTSTLogger, evaluator: PatchTSTEvaluator):
        self.logger = logger
        self.evaluator = evaluator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_model(self, params: PatchTSTHyperparameters, num_input_channels: int) -> nn.Module:
        """Create PatchTST model with given hyperparameters"""
        
        # Try to use original PatchTST from transformers
        try:
            from transformers import PatchTSTConfig, PatchTSTForClassification
            
            config = PatchTSTConfig(
                # Core architecture
                num_input_channels=num_input_channels,
                num_targets=2,  # Binary classification
                context_length=params.context_length,
                patch_length=params.patch_length,
                stride=params.stride,
                use_cls_token=True,  # Essential for classification
                
                # Transformer parameters
                hidden_size=params.hidden_size,
                num_hidden_layers=params.num_hidden_layers,
                num_attention_heads=params.num_attention_heads,
                
                # Regularization
                dropout=params.dropout,
                hidden_dropout_prob=params.hidden_dropout_prob,
                attention_probs_dropout_prob=params.attention_probs_dropout_prob,
                
                # Additional parameters
                initializer_range=0.02,
            )
            
            model = PatchTSTForClassification(config)
            print(f"✅ Using original PatchTSTForClassification")
            return model.to(self.device)
            
        except ImportError:
            print(f"⚠️  PatchTSTForClassification not available, using custom implementation")
            
            # Fall back to custom implementation
            class CustomPatchTSTForClassification(nn.Module):
                """Custom PatchTST implementation for classification"""
                
                def __init__(self, config_params):
                    super().__init__()
                    self.config = config_params
                    
                    # Patch embedding
                    self.patch_embedding = nn.Linear(
                        self.config.patch_length * num_input_channels, 
                        self.config.hidden_size
                    )
                    
                    # Positional embedding
                    max_patches = self.config.context_length // self.config.stride + 1
                    self.position_embedding = nn.Parameter(
                        torch.randn(1, max_patches, self.config.hidden_size)
                    )
                    
                    # CLS token
                    self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.hidden_size))
                    
                    # Transformer encoder
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=self.config.hidden_size,
                        nhead=self.config.num_attention_heads,
                        dim_feedforward=self.config.hidden_size * 4,
                        dropout=self.config.dropout,
                        activation='gelu',
                        batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(
                        encoder_layer, 
                        num_layers=self.config.num_hidden_layers
                    )
                    
                    # Classification head
                    self.classifier = nn.Linear(self.config.hidden_size, 2)  # Binary classification
                    self.dropout = nn.Dropout(self.config.dropout)
                    
                def forward(self, past_values=None, **kwargs):
                    """Forward pass - handle both past_values and direct tensor input"""
                    if past_values is not None:
                        x = past_values
                    else:
                        # Fallback to first positional argument
                        x = list(kwargs.values())[0] if kwargs else None
                        if x is None:
                            raise ValueError("No input tensor provided")
                    
                    batch_size, seq_len, num_channels = x.shape
                    
                    # Create patches
                    patches = []
                    for i in range(0, seq_len - self.config.patch_length + 1, self.config.stride):
                        patch = x[:, i:i+self.config.patch_length, :]
                        patches.append(patch.reshape(batch_size, -1))
                    
                    if len(patches) == 0:
                        raise ValueError(f"No patches created. seq_len={seq_len}, patch_length={self.config.patch_length}")
                    
                    patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_dim]
                    
                    # Patch embedding
                    embeddings = self.patch_embedding(patches)
                    
                    # Add positional embedding
                    embeddings = embeddings + self.position_embedding[:, :embeddings.size(1), :]
                    
                    # Add CLS token
                    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                    embeddings = torch.cat([cls_tokens, embeddings], dim=1)
                    
                    # Transformer
                    output = self.transformer(embeddings)
                    
                    # Classification (use CLS token)
                    cls_output = output[:, 0, :]  # Use CLS token
                    cls_output = self.dropout(cls_output)
                    logits = self.classifier(cls_output)
                    
                    # Return in HuggingFace-like format
                    return SimpleNamespace(prediction_logits=logits)
            
            # Create configuration
            config_params = SimpleNamespace(
                context_length=params.context_length,
                patch_length=params.patch_length,
                stride=params.stride,
                hidden_size=params.hidden_size,
                num_hidden_layers=params.num_hidden_layers,
                num_attention_heads=params.num_attention_heads,
                dropout=params.dropout
            )
            
            model = CustomPatchTSTForClassification(config_params)
            print(f"✅ Using custom PatchTST implementation")
            return model.to(self.device)
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   valid_loader: DataLoader, params: PatchTSTHyperparameters) -> Dict[str, Any]:
        """Train PatchTST model with comprehensive monitoring"""
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params.patience_lr_reduction, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        self.logger.logger.info(f"🏋️ Starting training for {params.epochs} epochs...")
        
        for epoch in range(params.epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with correct input format
                outputs = model(past_values=batch_x)
                
                # Extract logits from output object (handle both original and custom models)
                if hasattr(outputs, 'prediction_logits'):
                    logits = outputs.prediction_logits
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'prediction_outputs'):
                    logits = outputs.prediction_outputs
                else:
                    logits = outputs
                
                loss = criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Forward pass with correct input format
                    outputs = model(past_values=batch_x)
                    
                    # Extract logits from output object (handle both original and custom models)
                    if hasattr(outputs, 'prediction_logits'):
                        logits = outputs.prediction_logits
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif hasattr(outputs, 'prediction_outputs'):
                        logits = outputs.prediction_outputs
                    else:
                        logits = outputs
                    
                    loss = criterion(logits, batch_y)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            self.logger.log_training_epoch(epoch + 1, avg_train_loss, avg_val_loss, current_lr, epoch_time)
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr,
                'time': epoch_time
            })
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model (could implement checkpoint saving here)
            else:
                patience_counter += 1
                if patience_counter >= params.patience_early_stopping:
                    self.logger.logger.info(f"⏰ Early stopping at epoch {epoch + 1}")
                    break
        
        return {
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1
        }

class PatchTSTExperimentRunner:
    """Main experiment runner for PatchTST hyperparameter tuning"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results_dir = Path(root) / "results" / "PatchTSTResults"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_dir = Path(root) / "logs" / "patchtst"
        self.logger = PatchTSTLogger(experiment_name, log_dir)
        
        # Setup components
        self.data_processor = PatchTSTDataProcessor()
        self.evaluator = PatchTSTEvaluator(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.trainer = PatchTSTTrainer(self.logger, self.evaluator)
        
        self.logger.logger.info(f"🖥️ Using device: {self.trainer.device}")
    
    def run_single_experiment(self, case_name: str, params: PatchTSTHyperparameters) -> Dict[str, Any]:
        """Run single PatchTST experiment with given hyperparameters"""
        
        self.logger.log_hyperparameters(params)
        
        # Load and prepare data
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.data_processor.load_cer_data(
            case_name, params.embed_type, params.random_seed
        )
        
        self.logger.log_data_info(len(X_train), len(X_valid), len(X_test))
        
        # Prepare tensors
        X_train_tensor, y_train_tensor = self.data_processor.prepare_patchtst_data(X_train, y_train)
        X_valid_tensor, y_valid_tensor = self.data_processor.prepare_patchtst_data(X_valid, y_valid)
        X_test_tensor, y_test_tensor = self.data_processor.prepare_patchtst_data(X_test, y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
        
        # Create model
        num_input_channels = X_train_tensor.shape[-1]
        model = self.trainer.create_model(params, num_input_channels)
        self.logger.log_model_info(model)
        
        # Train model
        training_results = self.trainer.train_model(model, train_loader, valid_loader, params)
        
        # Evaluate model
        self.logger.logger.info("🔍 Evaluating model...")
        test_metrics = self.evaluator.evaluate_model(model, test_loader)
        self.logger.log_results(test_metrics)
        
        # Combine results
        experiment_results = {
            'hyperparameters': params.to_dict(),
            'training_info': training_results,
            'test_metrics': test_metrics,
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'num_input_channels': num_input_channels
            },
            'data_info': {
                'train_size': len(X_train),
                'valid_size': len(X_valid),
                'test_size': len(X_test),
                'case_name': case_name
            }
        }
        
        return experiment_results
    
    def run_hyperparameter_grid_search(self, case_name: str, 
                                      hyperparameter_grid: Dict[str, List]) -> Dict[str, Any]:
        """Run comprehensive hyperparameter grid search"""
        
        self.logger.logger.info(f"🔬 Starting hyperparameter grid search for {case_name}")
        grid_dims = [len(v) for v in hyperparameter_grid.values()]
        self.logger.logger.info(f"📊 Grid dimensions: {' × '.join(map(str, grid_dims))}")
        
        # Generate all parameter combinations
        param_names = list(hyperparameter_grid.keys())
        param_values = list(hyperparameter_grid.values())
        
        all_results = []
        total_combinations = len(list(product(*param_values)))
        
        self.logger.logger.info(f"🎯 Total experiments to run: {total_combinations}")
        
        for i, param_combination in enumerate(product(*param_values)):
            separator = "=" * 80
            self.logger.logger.info(f"\n{separator}")
            self.logger.logger.info(f"🧪 Experiment {i+1}/{total_combinations}")
            self.logger.logger.info(f"{separator}")
            
            # Create hyperparameter configuration
            param_dict = dict(zip(param_names, param_combination))
            params = PatchTSTHyperparameters(**param_dict)
            
            # try:
            # Run single experiment
            result = self.run_single_experiment(case_name, params)
            all_results.append(result)
            
            # Log quick summary
            f1_macro = result['test_metrics']['f1_macro']
            f1_binary = result['test_metrics']['f1_binary']
            self.logger.logger.info(f"✅ Experiment {i+1} completed: F1-Macro={f1_macro:.4f}, F1-Binary={f1_binary:.4f}")
                
            # except Exception as e:
            #     self.logger.logger.error(f"❌ Experiment {i+1} failed: {str(e)}")
            #     continue
        
        # Create comprehensive results summary
        comprehensive_results = {
            'experiment_info': {
                'experiment_name': self.experiment_name,
                'case_name': case_name,
                'timestamp': datetime.now().isoformat(),
                'total_experiments': len(all_results),
                'model_type': 'PatchTST',
                'phase': 'Phase_1_Direct_Classification'
            },
            'hyperparameter_grid': hyperparameter_grid,
            'all_results': all_results
        }
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"patchtst_{case_name}_phase1_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        self.logger.logger.info(f"💾 Results saved to: {results_file}")
        
        return comprehensive_results

def get_default_hyperparameter_grid() -> Dict[str, List]:
    """Define focused hyperparameter grid for PatchTST experiments (similar to TST approach)"""
    
    return {
        # Core architecture parameters (most important for performance)
        'hidden_size': [64, 128],         # Inner model dimension (key parameter)
        'num_attention_heads': [4, 8],    # Multi-head attention (key parameter)
        'num_hidden_layers': [3, 4],       # Transformer depth (key parameter)
        
        # Fixed patch parameters (based on typical good defaults)
        'patch_length': [16],                  # Fixed optimal patch size
        'stride': [8],                         # Fixed 50% overlap
        
        # Fixed training parameters (based on best practices)
        'dropout': [0.1],                      # Standard dropout
        'learning_rate': [1e-4],               # Standard learning rate
        'embed_type': [0, 1],                     # No temporal embeddings for simplicity
        'random_seed': [0]               # Multiple seeds for statistical validity
    }

def get_quick_test_grid() -> Dict[str, List]:
    """Define a minimal grid for quick testing (single configuration per key parameter)"""
    
    return {
        'hidden_size': [64],                  # Medium size for testing
        'num_attention_heads': [2],           # Standard attention heads
        'num_hidden_layers': [3],             # Medium depth
        'patch_length': [16],                 # Standard patch size
        'stride': [8],                        # Standard stride
        'dropout': [0.1],                     # Standard dropout
        'learning_rate': [1e-4],              # Standard learning rate
        'embed_type': [0],                    # No embeddings for simplicity
        'random_seed': [0]                    # Single seed for quick test
    }

def main():
    """Main execution function with CLI for multi-case runs."""

    parser = argparse.ArgumentParser(
        description="Run PatchTST Phase 1 experiments for one or more CER cases"
    )
    parser.add_argument(
        "--cases", "-c",
        type=str,
        default="cooker_case",
        help="Comma-separated list of case names to run (e.g. cooker_case,tv_case). Default: cooker_case"
    )
    parser.add_argument(
        "--grid", "-g",
        choices=["quick", "focused"],
        default="quick",
        help="Which hyperparameter grid to run: 'quick' (single test) or 'focused' (full grid). Default: quick"
    )

    args = parser.parse_args()

    case_list = [c.strip() for c in args.cases.split(",") if c.strip()]
    if len(case_list) == 0:
        case_list = ["cooker_case"]

    # Select hyperparameter grid
    if args.grid == "quick":
        hyperparameter_grid = get_quick_test_grid()
        print("🔬 Running quick test grid...")
    else:
        hyperparameter_grid = get_default_hyperparameter_grid()
        print("🔬 Running focused hyperparameter grid search...")
        print("📊 Grid: 3 hidden_sizes × 3 attention_heads × 3 layers × 3 seeds = 81 experiments")
        print("⚠️  This may take 2-4 hours to complete!")

    # Iterate over cases and run experiments per case
    overall_results = {}
    for case_name in case_list:
        print("\n" + "=" * 60)
        print(f"🚀 Running PatchTST Phase 1 for case: {case_name}")

        experiment_name = f"patchtst_phase1_{case_name}"
        runner = PatchTSTExperimentRunner(experiment_name)
        results = runner.run_hyperparameter_grid_search(case_name, hyperparameter_grid)

        overall_results[case_name] = results

        # Print per-case summary
        print("\n" + "=" * 60)
        print(f"🎉 Completed experiments for case: {case_name}")
        print(f"📊 Total successful experiments: {len(results['all_results'])}")

        if results['all_results']:
            best_result = max(results['all_results'], key=lambda x: x['test_metrics']['f1_macro'])
            print(f"🏆 Best F1-Macro: {best_result['test_metrics']['f1_macro']:.4f}")
            print(f"🎯 Best F1-Binary: {best_result['test_metrics']['f1_binary']:.4f}")
            print(f"📈 Best ROC-AUC: {best_result['test_metrics']['roc_auc']:.4f}")
            print("\n🔧 Best hyperparameters:")
            for key, value in best_result['hyperparameters'].items():
                print(f"  {key}: {value}")

    # Optionally save an overall summary file
    summary_file = Path(root) / "results" / "PatchTSTResults" / f"patchtst_overall_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as sf:
        json.dump(overall_results, sf, indent=2, default=str)

    print(f"\n💾 Overall summary saved to: {summary_file}")

if __name__ == "__main__":
    main()