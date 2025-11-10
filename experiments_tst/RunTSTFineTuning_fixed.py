#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Vansh Dhar (Based on TransApp framework)
# @description : Comprehensive TST Fine-Tuning Experiments with Pre-trained Models
# @component: experiments_tst/
# @file : RunTSTFineTuning_fixed.py
#
#################################################################################################################

import os, sys
import numpy as np
import pandas as pd
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Fix path resolution - get to TransApp root directory
current_file = Path(__file__).resolve()
root = current_file.parents[1]  # Go up from experiments_tst/ to TransApp/
sys.path.insert(0, str(root))  # Insert at beginning of path

# Import TransApp components
try:
    from experiments.data_utils import *
    from src.TransAppModel.TransApp_TST import *
    from src.AD_Framework.Framework import *
    from src.utils.losses import *
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Root directory: {root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

def setup_logging(log_file: str):
    """Setup comprehensive logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_pretrained_model(pretrained_path: str, model_config: dict, logger: logging.Logger) -> Optional[nn.Module]:
    """
    Load pre-trained TST model with proper error handling
    
    Args:
        pretrained_path: Path to pre-trained model checkpoint
        model_config: Model configuration dictionary
        logger: Logger instance
    
    Returns:
        Loaded model or None if loading fails
    """
    try:
        if not os.path.exists(pretrained_path):
            logger.error(f"❌ Pre-trained model not found: {pretrained_path}")
            return None
        
        logger.info(f"🔄 Loading pre-trained model from: {pretrained_path}")
        
        # Create model architecture
        model = TransApp_TST(
            c_in=model_config['c_in'],
            c_out=model_config['c_out'], 
            seq_len=model_config['seq_len'],
            d_model=model_config['d_model'],
            d_ff=model_config.get('d_ff', model_config['d_model'] * 4),
            n_heads=model_config.get('n_heads', 8),
            n_layers=model_config.get('n_layers', 3),
            norm_type=model_config.get('norm_type', 'BatchNorm'),
            embed_type=model_config.get('embed_type', 0),
            exo_variables=model_config.get('exo_variables', [])
        )
        
        # Load pre-trained weights
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ Loaded model from checkpoint with metadata")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"✅ Loaded model state dict directly")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"✅ Loaded model weights successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load pre-trained model: {e}")
        return None

def evaluate_quantile_based(model: nn.Module, 
                           X_test_voter: np.ndarray, 
                           y_test_voter: np.ndarray,
                           logger: logging.Logger) -> dict:
    """
    Evaluate model using quantile-based approach on full time series
    
    Args:
        model: Trained model
        X_test_voter: Full time series test data
        y_test_voter: Full time series test labels
        logger: Logger instance
    
    Returns:
        Dictionary containing quantile-based evaluation metrics
    """
    
    try:
        # Set model to evaluation mode
        model.eval()
        device = next(model.parameters()).device
        
        logger.info(f"🔍 Running quantile-based evaluation...")
        
        # Prepare data for quantile evaluation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test_voter.reshape(-1, X_test_voter.shape[-1])).reshape(X_test_voter.shape)
        
        scores = []
        with torch.no_grad():
            for i in range(X_test_voter.shape[0]):
                # Get time series for current sample
                x_sample = torch.FloatTensor(X_scaled[i:i+1]).to(device)
                
                # Get model predictions for all subsequences
                pred = model(x_sample)
                
                # Convert to probabilities and get positive class scores
                if pred.shape[-1] == 2:  # Binary classification
                    prob = torch.softmax(pred, dim=-1)[:, 1]  # Positive class probability
                else:
                    prob = torch.sigmoid(pred.squeeze())  # Single output with sigmoid
                
                # Use 90th percentile as final score (quantile-based approach)
                score = torch.quantile(prob, 0.9).cpu().item()
                scores.append(score)
        
        scores = np.array(scores)
        
        # Find optimal threshold
        from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, f1_score
        
        precision, recall, thresholds = precision_recall_curve(y_test_voter, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
        # Generate predictions with best threshold
        y_pred = (scores >= best_threshold).astype(int)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test_voter, y_pred)
        f1_macro = f1_score(y_test_voter, y_pred, average='macro')
        f1_weighted = f1_score(y_test_voter, y_pred, average='weighted')
        f1_binary = f1_score(y_test_voter, y_pred, average='binary')
        
        try:
            roc_auc = roc_auc_score(y_test_voter, scores)
        except:
            roc_auc = 0.5  # Default for cases with single class
        
        # Detailed classification report
        class_report = classification_report(y_test_voter, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test_voter, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_binary': f1_binary,
            'roc_auc': roc_auc,
            'best_threshold': best_threshold,
            'precision_macro': class_report['macro avg']['precision'],
            'recall_macro': class_report['macro avg']['recall'],
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        logger.info(f"📊 Quantile Evaluation Results:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   F1-Score (Macro): {f1_macro:.4f}")
        logger.info(f"   F1-Score (Binary): {f1_binary:.4f}")
        logger.info(f"   ROC-AUC: {roc_auc:.4f}")
        logger.info(f"   Best Threshold: {best_threshold:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Quantile evaluation failed: {e}")
        return {}

def fine_tune_model(model: nn.Module, 
                   datas_tuple: tuple,
                   fine_tune_params: dict,
                   case_name: str,
                   save_path: str,
                   logger: logging.Logger) -> dict:
    """
    Fine-tune pre-trained TST model on specific appliance case
    
    Args:
        model: Pre-trained TST model
        datas_tuple: Tuple of training/validation/test data
        fine_tune_params: Fine-tuning hyperparameters
        case_name: Name of appliance case
        save_path: Path to save fine-tuned model
        logger: Logger instance
    
    Returns:
        Dictionary containing training results and metrics
    """
    
    logger.info(f"🚀 Starting fine-tuning for {case_name}")
    logger.info(f"📋 Fine-tuning parameters: {fine_tune_params}")
    
    start_time = time.time()
    
    # Unpack data
    X_train, y_train = datas_tuple[0], datas_tuple[1]
    X_valid, y_valid = datas_tuple[2], datas_tuple[3]
    X_test, y_test = datas_tuple[4], datas_tuple[5]
    X_test_voter, y_test_voter = datas_tuple[10], datas_tuple[11]
    
    logger.info(f"📊 Data shapes:")
    logger.info(f"   Training: {X_train.shape}, {y_train.shape}")
    logger.info(f"   Validation: {X_valid.shape}, {y_valid.shape}")
    logger.info(f"   Test: {X_test.shape}, {y_test.shape}")
    logger.info(f"   Test voter: {X_test_voter.shape}, {y_test_voter.shape}")
    
    # Prepare datasets
    train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
    valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
    test_dataset = TSDataset(X_test, y_test, scaler=True, scale_dim=[0])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=fine_tune_params['batch_size'], 
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=1, 
        shuffle=False
    )
    
    # Setup AD Framework for fine-tuning
    model_trainer = AD_Framework(
        model,
        train_loader=train_loader, 
        valid_loader=valid_loader,
        learning_rate=fine_tune_params['lr'], 
        weight_decay=fine_tune_params['wd'],
        name_scheduler='ReduceLROnPlateau',
        max_epochs=fine_tune_params['epochs'],
        patience_early_stopping=fine_tune_params['patience_es'],
        patience_reduce_lr=fine_tune_params['patience_rlr'],
        factor_reduce_lr=0.5,
        min_lr=1e-6,
        verbose=True
    )
    
    logger.info(f"🏋️ Starting fine-tuning with {fine_tune_params['epochs']} epochs...")
    
    # Fine-tune the model
    try:
        model_trainer.fit()
        training_time = time.time() - start_time
        logger.info(f"⏱️ Fine-tuning completed in {training_time:.2f} seconds")
        
        # Evaluate on test set (subsequences)
        logger.info(f"📊 Evaluating on subsequence test set...")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        subsequence_score, subsequence_metrics = model_trainer.evaluate(test_loader)
        logger.info(f"📈 Subsequence F1-Score: {subsequence_score:.4f}")
        
        # Evaluate on full time series (quantile-based)
        logger.info(f"🔍 Evaluating on full time series with quantile method...")
        quantile_metrics = evaluate_quantile_based(model, X_test_voter, y_test_voter, logger)
        
        # Save fine-tuned model
        model_save_path = os.path.join(save_path, f"finetuned_{case_name}_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'fine_tune_params': fine_tune_params,
            'case_name': case_name,
            'training_time': training_time,
            'subsequence_score': subsequence_score,
            'quantile_metrics': quantile_metrics
        }, model_save_path)
        
        logger.info(f"💾 Fine-tuned model saved to: {model_save_path}")
        
        return {
            'case_name': case_name,
            'training_time': training_time,
            'subsequence_score': subsequence_score,
            'subsequence_metrics': subsequence_metrics,
            'quantile_metrics': quantile_metrics,
            'model_path': model_save_path,
            'fine_tune_params': fine_tune_params
        }
        
    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {e}")
        return None

def run_comprehensive_fine_tuning_experiments(
    pretrained_models: List[str],
    appliance_cases: List[str],
    fine_tune_configs: List[dict],
    output_dir: str,
    logger: logging.Logger
) -> dict:
    """
    Run comprehensive fine-tuning experiments across all configurations
    """
    
    logger.info(f"🚀 Starting comprehensive fine-tuning experiments...")
    logger.info(f"📋 Pre-trained models: {len(pretrained_models)}")
    logger.info(f"📋 Appliance cases: {len(appliance_cases)}")
    logger.info(f"📋 Fine-tuning configs: {len(fine_tune_configs)}")
    
    all_results = []
    total_experiments = len(pretrained_models) * len(appliance_cases) * len(fine_tune_configs)
    experiment_count = 0
    
    for pretrained_path in pretrained_models:
        for case_name in appliance_cases:
            for config in fine_tune_configs:
                
                experiment_count += 1
                logger.info(f"\n🧪 Experiment {experiment_count}/{total_experiments}")
                logger.info(f"   Model: {os.path.basename(pretrained_path)}")
                logger.info(f"   Case: {case_name}")
                logger.info(f"   Config: {config['name']}")
                
                try:
                    # Load data for this case
                    logger.info(f"📥 Loading data for {case_name}...")
                    datas_tuple = load_cer_case_data(case_name, config['dataset_type'])
                    
                    if datas_tuple is None:
                        logger.warning(f"⚠️ Could not load data for {case_name}, skipping...")
                        continue
                    
                    # Extract model configuration from pretrained path
                    model_config = extract_model_config_from_path(pretrained_path)
                    if model_config is None:
                        logger.warning(f"⚠️ Could not extract model config from {pretrained_path}, skipping...")
                        continue
                    
                    # Load pre-trained model
                    model = load_pretrained_model(pretrained_path, model_config, logger)
                    if model is None:
                        logger.warning(f"⚠️ Could not load pre-trained model, skipping...")
                        continue
                    
                    # Create save directory for this experiment
                    exp_save_dir = os.path.join(
                        output_dir, 
                        f"{case_name}_{config['name']}_{os.path.basename(pretrained_path).split('.')[0]}"
                    )
                    os.makedirs(exp_save_dir, exist_ok=True)
                    
                    # Run fine-tuning
                    result = fine_tune_model(
                        model=model,
                        datas_tuple=datas_tuple,
                        fine_tune_params=config['params'],
                        case_name=case_name,
                        save_path=exp_save_dir,
                        logger=logger
                    )
                    
                    if result is not None:
                        result.update({
                            'experiment_id': experiment_count,
                            'pretrained_model_path': pretrained_path,
                            'config_name': config['name'],
                            'timestamp': datetime.now().isoformat()
                        })
                        all_results.append(result)
                        logger.info(f"✅ Experiment {experiment_count} completed successfully")
                    else:
                        logger.error(f"❌ Experiment {experiment_count} failed")
                
                except Exception as e:
                    logger.error(f"❌ Experiment {experiment_count} failed with error: {e}")
                    continue
    
    # Save comprehensive results
    results_summary = {
        'total_experiments': total_experiments,
        'successful_experiments': len(all_results),
        'timestamp': datetime.now().isoformat(),
        'results': all_results
    }
    
    results_file = os.path.join(output_dir, f"comprehensive_finetuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"\n🏁 Comprehensive experiments completed!")
    logger.info(f"📊 Results saved to: {results_file}")
    logger.info(f"✅ Successful experiments: {len(all_results)}/{total_experiments}")
    
    return results_summary

def extract_model_config_from_path(model_path: str) -> Optional[dict]:
    """Extract model configuration from pre-trained model path"""
    
    try:
        filename = os.path.basename(model_path)
        
        # Parse filename like: TransApp_TST_64_BatchNorm.pt
        parts = filename.replace('.pt', '').split('_')
        
        if len(parts) >= 4:
            model_type = '_'.join(parts[:2])  # TransApp_TST
            d_model = int(parts[2])  # 64
            norm_type = parts[3]  # BatchNorm
            
            # Default configuration based on TST architecture
            config = {
                'c_in': 1,  # Single variable (consumption)
                'c_out': 2,  # Binary classification
                'seq_len': 2016,  # Standard window size
                'd_model': d_model,
                'd_ff': d_model * 4,
                'n_heads': 8,
                'n_layers': 3,
                'norm_type': norm_type,
                'embed_type': 0,  # No exogenous variables by default
                'exo_variables': []
            }
            
            return config
    
    except Exception as e:
        print(f"❌ Error extracting config from {model_path}: {e}")
    
    return None

def load_cer_case_data(case_name: str, dataset_type: str = "CER") -> Optional[tuple]:
    """Load CER dataset for specific appliance case"""
    
    try:
        # Load CER data following the same methodology as RunTransAppClassif_TST.py
        embed_type = 0  # No exogenous variables
        exo_variables = []
        
        # Use the existing data loading utilities
        datas_tuple = load_cer(case_name, embed_type, exo_variables, dataset_type)
        
        return datas_tuple
        
    except Exception as e:
        print(f"❌ Error loading data for {case_name}: {e}")
        return None

def generate_summary_report(results: dict, output_dir: str, logger: logging.Logger):
    """Generate comprehensive summary report of fine-tuning results"""
    
    try:
        logger.info(f"📊 Generating summary report...")
        
        # Create summary DataFrame
        import pandas as pd
        
        summary_data = []
        for result in results['results']:
            if result.get('quantile_metrics'):
                summary_data.append({
                    'case_name': result['case_name'],
                    'config_name': result['config_name'],
                    'pretrained_model': os.path.basename(result['pretrained_model_path']),
                    'training_time': result['training_time'],
                    'subsequence_f1': result['subsequence_score'],
                    'quantile_accuracy': result['quantile_metrics']['accuracy'],
                    'quantile_f1_macro': result['quantile_metrics']['f1_macro'],
                    'quantile_f1_binary': result['quantile_metrics']['f1_binary'],
                    'quantile_roc_auc': result['quantile_metrics']['roc_auc'],
                    'best_threshold': result['quantile_metrics']['best_threshold']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Save detailed CSV
            csv_file = os.path.join(output_dir, f"finetuning_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(csv_file, index=False)
            logger.info(f"📄 Summary CSV saved to: {csv_file}")
            
            # Generate summary statistics
            logger.info(f"\n📊 FINE-TUNING RESULTS SUMMARY")
            logger.info(f"{'='*60}")
            
            # Overall statistics
            logger.info(f"📈 Overall Performance:")
            logger.info(f"   Mean F1-Score (Macro): {df['quantile_f1_macro'].mean():.4f} ± {df['quantile_f1_macro'].std():.4f}")
            logger.info(f"   Mean F1-Score (Binary): {df['quantile_f1_binary'].mean():.4f} ± {df['quantile_f1_binary'].std():.4f}")
            logger.info(f"   Mean Accuracy: {df['quantile_accuracy'].mean():.4f} ± {df['quantile_accuracy'].std():.4f}")
            logger.info(f"   Mean ROC-AUC: {df['quantile_roc_auc'].mean():.4f} ± {df['quantile_roc_auc'].std():.4f}")
            
            # Best results per case
            logger.info(f"\n🏆 Best Results by Appliance Case:")
            for case in df['case_name'].unique():
                case_df = df[df['case_name'] == case]
                best_idx = case_df['quantile_f1_macro'].idxmax()
                best_result = case_df.loc[best_idx]
                
                logger.info(f"   {case}:")
                logger.info(f"     F1-Macro: {best_result['quantile_f1_macro']:.4f}")
                logger.info(f"     Config: {best_result['config_name']}")
                logger.info(f"     Model: {best_result['pretrained_model']}")
            
            # Best results per configuration
            logger.info(f"\n⚙️ Best Results by Configuration:")
            for config in df['config_name'].unique():
                config_df = df[df['config_name'] == config]
                logger.info(f"   {config}:")
                logger.info(f"     Mean F1-Macro: {config_df['quantile_f1_macro'].mean():.4f} ± {config_df['quantile_f1_macro'].std():.4f}")
                logger.info(f"     Best F1-Macro: {config_df['quantile_f1_macro'].max():.4f}")
                logger.info(f"     Mean Training Time: {config_df['training_time'].mean():.2f}s")
        
        logger.info(f"✅ Summary report generated successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate summary report: {e}")

def main():
    """Main function to run TST fine-tuning experiments"""
    
    parser = argparse.ArgumentParser(description='TST Fine-Tuning Experiments')
    parser.add_argument('--pretrained_dir', type=str, 
                       default='results/TransAppPretrained_TST',
                       help='Directory containing pre-trained models')
    parser.add_argument('--output_dir', type=str, 
                       default='results/TST_FineTuning',
                       help='Directory to save fine-tuning results')
    parser.add_argument('--cases', nargs='+', 
                       default=['cooker_case', 'dishwasher_case', 'waterheater_case', 
                               'tumbledryer_case', 'tv_greater21inch_case'],
                       help='List of appliance cases to test')
    parser.add_argument('--dataset_type', type=str, default='CER',
                       choices=['CER', 'COMSTOCK'],
                       help='Dataset type to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, f"tst_finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file)
    
    logger.info(f"🚀 Starting TST Fine-Tuning Experiments")
    logger.info(f"📂 Pre-trained models directory: {args.pretrained_dir}")
    logger.info(f"📂 Output directory: {args.output_dir}")
    logger.info(f"📋 Appliance cases: {args.cases}")
    logger.info(f"📊 Dataset type: {args.dataset_type}")
    
    # Find pre-trained models
    pretrained_models = []
    for root_dir, dirs, files in os.walk(args.pretrained_dir):
        for file in files:
            if file.endswith('.pt') and 'TransApp_TST' in file:
                pretrained_models.append(os.path.join(root_dir, file))
    
    if not pretrained_models:
        logger.error(f"❌ No pre-trained TST models found in {args.pretrained_dir}")
        return
    
    logger.info(f"🔍 Found {len(pretrained_models)} pre-trained models:")
    for model in pretrained_models:
        logger.info(f"   📁 {model}")
    
    # Define fine-tuning configurations
    fine_tune_configs = [
        {
            'name': 'conservative',
            'dataset_type': args.dataset_type,
            'params': {
                'lr': 1e-5,  # Lower learning rate for fine-tuning
                'wd': 1e-4,
                'batch_size': 16,
                'epochs': 10,
                'patience_es': 5,
                'patience_rlr': 3
            }
        },
        {
            'name': 'moderate',
            'dataset_type': args.dataset_type,
            'params': {
                'lr': 5e-5,
                'wd': 1e-3,
                'batch_size': 16,
                'epochs': 15,
                'patience_es': 5,
                'patience_rlr': 3
            }
        },
        {
            'name': 'aggressive',
            'dataset_type': args.dataset_type,
            'params': {
                'lr': 1e-4,
                'wd': 1e-3,
                'batch_size': 32,
                'epochs': 20,
                'patience_es': 7,
                'patience_rlr': 4
            }
        }
    ]
    
    logger.info(f"⚙️ Fine-tuning configurations:")
    for config in fine_tune_configs:
        logger.info(f"   🔧 {config['name']}: LR={config['params']['lr']}, Epochs={config['params']['epochs']}")
    
    # Run comprehensive experiments
    results = run_comprehensive_fine_tuning_experiments(
        pretrained_models=pretrained_models,
        appliance_cases=args.cases,
        fine_tune_configs=fine_tune_configs,
        output_dir=args.output_dir,
        logger=logger
    )
    
    # Generate summary report
    generate_summary_report(results, args.output_dir, logger)
    
    logger.info(f"🏁 TST Fine-Tuning Experiments Completed Successfully!")

if __name__ == "__main__":
    main()