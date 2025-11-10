#!/usr/bin/env python3
"""
Comprehensive DMSA-TST Hybrid Experiment
========================================

This script tests the innovative hybrid approach combining TST architecture 
with optional DMSA (Diagonally Masked Self-Attention) to determine:

1. Standard TST vs DMSA-TST performance comparison
2. Impact of diagonal masking on appliance detection
3. Best configuration for each attention mechanism

Author: Enhanced TransApp Framework
Date: October 2025
"""

import sys
import os
import json
import torch
import torch.utils.data
import numpy as np
from datetime import datetime
from pathlib import Path

# Fix path resolution - get to TransApp root directory
current_file = Path(__file__).resolve()
root = current_file.parents[1]  # Go up from experiments_tst/ to TransApp/
sys.path.insert(0, str(root))  # Insert at beginning of path

# Import TransApp modules
try:
    from experiments.data_utils import CER_get_data_case, COMSTOCK_get_data_case
    from src.TransAppModel.TransApp_TST_DMSA import TransApp_TST_DMSA
    from src.AD_Framework.Framework import TSDataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Root directory: {root}")
    sys.exit(1)

def load_data_for_dmsa_experiment(case_name, embed_type, exo_variables, random_seed, dataset_type="CER"):
    """Load data for DMSA-TST experiment"""
    
    if dataset_type == "CER":
        # Use CER data loading function
        datas_tuple = CER_get_data_case(
            case_name=case_name, 
            seed=random_seed, 
            exo_variable=exo_variables,
            win=96,  # 15-minute intervals over 24 hours
            ratio_resample=0.8
        )
    else:
        # Use COMSTOCK data loading function  
        datas_tuple = COMSTOCK_get_data_case(
            case_name=case_name,
            seed=random_seed,
            win=96
        )
    
    # Extract data from tuple
    X_train = datas_tuple[0]
    y_train = datas_tuple[1] 
    X_valid = datas_tuple[2]
    y_valid = datas_tuple[3]
    X_test = datas_tuple[4]
    y_test = datas_tuple[5]
    
    print(f"📊 Data loaded - Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    
    return (X_train, y_train, X_valid, y_valid, X_test, y_test)

def get_device():
    """Get available device with fallback to CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def load_pretrained_model(model_path, model_config, device):
    """Load pre-trained model weights"""
    try:
        # Create model with same config
        model = TransApp_TST_DMSA(**model_config).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"✅ Loaded pre-trained model from {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        return None

def create_hybrid_model(embed_type, dim_model, norm_type, mask_diag, device, verbose=True):
    """Create DMSA-TST hybrid model"""
    
    # Determine input channels based on embedding type
    if embed_type == 0:  # No embeddings
        c_in = 1  # Just power consumption
        exo_variables = []
    else:  # With embeddings  
        c_in = 5  # Power + 4 time features
        exo_variables = ["hours_cos", "hours_sin", "days_cos", "days_sin"]
    
    model_config = {
        'c_in': c_in,
        'c_out': 1,  # Binary classification
        'seq_len': 96,  # 15-minute intervals over 24 hours
        'd_model': dim_model,
        'n_heads': 8,
        'e_layers': 3,
        'd_ff': dim_model * 4,
        'dropout': 0.1,
        'mask_diag': mask_diag,  # DMSA toggle
        'verbose': verbose
    }
    
    model = TransApp_TST_DMSA(**model_config).to(device)
    
    return model, model_config, exo_variables

def train_model(model, train_loader, device, epochs=15, lr=1e-4, wd=1e-3):
    """Train the model"""
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_data in train_loader:
            # Handle TSDataset format
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
            else:
                batch_x = batch_data
                continue  # Skip if no labels
                
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, task="classification")
            loss = criterion(outputs.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}]    Train loss: {avg_loss:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Early stopping patience
                print(f"Early stopping at epoch {epoch+1}")
                break

def evaluate_model(model, test_loader, device):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle TSDataset format
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
            else:
                batch_x = batch_data
                continue  # Skip if no labels
                
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x, task="classification")
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    f1_weighted = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')[2]
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probabilities)
    except:
        roc_auc = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'ACCURACY': accuracy,
        'PRECISION': float(precision[1]) if len(precision) > 1 else 0.0,
        'RECALL': float(recall[1]) if len(recall) > 1 else 0.0,
        'PRECISION_MACRO': precision_macro,
        'RECALL_MACRO': recall_macro,
        'F1_SCORE': float(f1[1]) if len(f1) > 1 else 0.0,
        'F1_SCORE_MACRO': f1_macro,
        'F1_SCORE_WEIGHTED': f1_weighted,
        'ROC_AUC_SCORE': roc_auc,
        'CONFUSION_MATRIX': str(cm)
    }

def run_dmsa_tst_experiment(case_name, embed_type, dim_model, norm_type, mask_diag, 
                           random_seed, epochs=15, device=None):
    """Run single DMSA-TST experiment"""
    
    if device is None:
        device = get_device()
    
    print(f"\n🧬 Running {'DMSA-TST' if mask_diag else 'Standard TST'} Experiment")
    print(f"📊 Case: {case_name}, Embed: {embed_type}, Dim: {dim_model}, Seed: {random_seed}")
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create model
    model, model_config, exo_variables = create_hybrid_model(
        embed_type, dim_model, norm_type, mask_diag, device, verbose=True
    )
    
    # Load data
    print("📊 Loading training data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_for_dmsa_experiment(
        case_name=case_name,
        embed_type=embed_type,
        exo_variables=exo_variables,
        random_seed=random_seed,
        dataset_type="CER"
    )
    
    # Create TSDataset objects (same as other TST experiments)
    train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
    test_dataset = TSDataset(X_test, y_test, scaler=True, scale_dim=[0])
    
    print(f"✅ Data prepared - Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Train model
    print("🏋️ Training model...")
    train_model(model, train_loader, device, epochs=epochs)
    
    # Evaluate model
    print("🔍 Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    return {
        'attention_type': 'DMSA' if mask_diag else 'Standard',
        'mask_diag': mask_diag,
        'embed_type': embed_type,
        'random_seed': random_seed,
        'results': results,
        'config': {
            'embed_type': embed_type,
            'exo_variables': exo_variables,
            'random_seed': random_seed,
            'dim_model': dim_model,
            'norm_type': norm_type,
            'mask_diag': mask_diag,
            'epochs': epochs
        }
    }

def run_comprehensive_dmsa_comparison(case_name="cooker_case", dim_model=96, norm_type="BatchNorm", epochs=15):
    """Run comprehensive comparison between Standard TST and DMSA-TST"""
    
    device = get_device()
    print(f"🖥️ Using device: {device}")
    
    all_results = []
    
    # Test configurations: [embed_type, mask_diag, random_seeds]
    configurations = [
        (0, False, [0, 1, 2]),  # No embeddings, Standard TST
        (0, True, [0, 1, 2]),   # No embeddings, DMSA-TST  
        (1, False, [0, 1, 2]),  # With embeddings, Standard TST
        (1, True, [0, 1, 2]),   # With embeddings, DMSA-TST
    ]
    
    for embed_type, mask_diag, seeds in configurations:
        attention_name = "DMSA-TST" if mask_diag else "Standard TST"
        embed_name = "With Embeddings" if embed_type == 1 else "No Embeddings"
        
        print(f"\n{'='*80}")
        print(f"🧪 Testing {attention_name} + {embed_name}")
        print(f"{'='*80}")
        
        for seed in seeds:
            try:
                result = run_dmsa_tst_experiment(
                    case_name=case_name,
                    embed_type=embed_type,
                    dim_model=dim_model,
                    norm_type=norm_type,
                    mask_diag=mask_diag,
                    random_seed=seed,
                    epochs=epochs,
                    device=device
                )
                all_results.append(result)
                
                # Print quick result
                f1_macro = result['results']['F1_SCORE_MACRO']
                print(f"✅ {attention_name} ({embed_name}, seed {seed}): F1-macro = {f1_macro:.4f}")
                
            except Exception as e:
                print(f"❌ Failed experiment: {attention_name} ({embed_name}, seed {seed}): {e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dmsa_tst_comparison_{case_name}_{dim_model}_{norm_type}_{epochs}ep_{timestamp}.json"
    results_dir = Path("/home/user/vansh/ISP/TransApp/results/TransAppResults_TST")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / filename
    
    experiment_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment_info": {
            "case_name": case_name,
            "model_name": "DMSA_TST_Comparison",
            "dim_model": dim_model,
            "epochs": epochs,
            "norm_type": norm_type,
            "dataset_type": "CER"
        },
        "all_results": all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print_comparison_summary(all_results)
    
    return all_results

def print_comparison_summary(results):
    """Print summary comparing DMSA vs Standard attention"""
    print(f"\n{'='*80}")
    print("📊 DMSA-TST vs Standard TST COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Group results by configuration
    groups = {}
    for result in results:
        embed_type = result['embed_type']
        mask_diag = result['mask_diag']
        key = f"Embed{embed_type}_{'DMSA' if mask_diag else 'Standard'}"
        
        if key not in groups:
            groups[key] = []
        groups[key].append(result['results']['F1_SCORE_MACRO'])
    
    # Print comparison
    for key, scores in groups.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{key:20s}: F1-macro = {mean_score:.4f} ± {std_score:.4f}")
    
    # Find best configuration
    best_score = 0
    best_config = ""
    for key, scores in groups.items():
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_config = key
    
    print(f"\n🏆 Best Configuration: {best_config} (F1-macro: {best_score:.4f})")

if __name__ == "__main__":
    print("🧬 Starting DMSA-TST Hybrid Experiment")
    print("="*60)
    

    # Run comprehensive comparison - QUICK TEST VERSION
    results = run_comprehensive_dmsa_comparison(
        case_name="cooker_case",
        dim_model=96,
        norm_type="BatchNorm", 
        epochs=5 # Quick test with just 2 epochs
    )
    
    print("\n✅ DMSA-TST Hybrid Experiment Completed!")