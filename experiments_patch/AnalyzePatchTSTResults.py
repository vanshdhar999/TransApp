#!/usr/bin/env python3
"""
PatchTST Results Analysis Script
===============================

This script analyzes and visualizes results from PatchTST Phase 1 experiments.
It provides comprehensive analysis including:
- Best hyperparameter identification
- Performance comparison across cases
- Statistical significance testing
- Hyperparameter sensitivity analysis
- Result visualization and reporting

Usage:
    python AnalyzePatchTSTResults.py [--case CASE_NAME] [--output-dir OUTPUT_DIR]
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings('ignore')

# Set up paths
current_file = Path(__file__).resolve()
root = current_file.parents[1]
sys.path.insert(0, str(root))

class PatchTSTResultsAnalyzer:
    """Comprehensive analysis of PatchTST experimental results"""
    
    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all result files
        self.all_results = self.load_all_results()
        
    def load_all_results(self) -> Dict[str, Any]:
        """Load all PatchTST result files"""
        
        result_files = list(self.results_dir.glob("patchtst_*.json"))
        all_results = {}
        
        print(f"📊 Found {len(result_files)} result files")
        
        for file in result_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    case_name = data['experiment_info']['case_name']
                    all_results[case_name] = data
                    print(f"✅ Loaded results for {case_name}: {len(data['all_results'])} experiments")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                
        return all_results
    
    def create_performance_summary(self) -> pd.DataFrame:
        """Create comprehensive performance summary"""
        
        summary_data = []
        
        for case_name, case_results in self.all_results.items():
            for experiment in case_results['all_results']:
                params = experiment['hyperparameters']
                metrics = experiment['test_metrics']
                
                row = {
                    'case_name': case_name,
                    'patch_length': params['patch_length'],
                    'stride': params['stride'],
                    'hidden_size': params['hidden_size'],
                    'num_hidden_layers': params['num_hidden_layers'],
                    'num_attention_heads': params['num_attention_heads'],
                    'dropout': params['dropout'],
                    'learning_rate': params['learning_rate'],
                    'embed_type': params['embed_type'],
                    'random_seed': params['random_seed'],
                    'f1_macro': metrics['f1_macro'],
                    'f1_binary': metrics['f1_binary'],
                    'accuracy': metrics['accuracy'],
                    'roc_auc': metrics['roc_auc'],
                    'precision_macro': metrics['precision_macro'],
                    'recall_macro': metrics['recall_macro'],
                    'total_parameters': experiment['model_info']['total_parameters'],
                    'total_epochs': experiment['training_info']['total_epochs'],
                    'best_val_loss': experiment['training_info']['best_val_loss']
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def find_best_configurations(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Find best configurations for each case"""
        
        best_configs = {}
        
        for case_name in df['case_name'].unique():
            case_df = df[df['case_name'] == case_name]
            
            # Find best by F1-macro
            best_idx = case_df['f1_macro'].idxmax()
            best_row = case_df.loc[best_idx]
            
            # Calculate statistics across all experiments for this case
            stats = {
                'mean_f1_macro': case_df['f1_macro'].mean(),
                'std_f1_macro': case_df['f1_macro'].std(),
                'max_f1_macro': case_df['f1_macro'].max(),
                'min_f1_macro': case_df['f1_macro'].min(),
                'mean_accuracy': case_df['accuracy'].mean(),
                'mean_roc_auc': case_df['roc_auc'].mean(),
                'total_experiments': len(case_df)
            }
            
            best_configs[case_name] = {
                'best_config': best_row.to_dict(),
                'statistics': stats
            }
        
        return best_configs
    
    def analyze_hyperparameter_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze hyperparameter importance using correlation"""
        
        # Select numeric hyperparameters and performance metric
        hyperparams = ['patch_length', 'stride', 'hidden_size', 'num_hidden_layers', 
                      'num_attention_heads', 'dropout', 'learning_rate', 'embed_type']
        
        importance_scores = {}
        
        for param in hyperparams:
            if param in df.columns:
                correlation = abs(df[param].corr(df['f1_macro']))
                importance_scores[param] = correlation if not np.isnan(correlation) else 0.0
        
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def create_performance_plots(self, df: pd.DataFrame):
        """Create comprehensive performance visualization plots"""
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. F1-Macro distribution by case
        plt.subplot(3, 3, 1)
        df.boxplot(column='f1_macro', by='case_name', ax=plt.gca())
        plt.title('F1-Macro Score Distribution by Case')
        plt.xlabel('Case')
        plt.ylabel('F1-Macro Score')
        plt.xticks(rotation=45)
        
        # 2. Hyperparameter correlation heatmap
        plt.subplot(3, 3, 2)
        hyperparams = ['patch_length', 'stride', 'hidden_size', 'num_hidden_layers', 
                      'num_attention_heads', 'dropout', 'learning_rate']
        corr_data = df[hyperparams + ['f1_macro']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=plt.gca())
        plt.title('Hyperparameter Correlation Matrix')
        
        # 3. Performance vs Model Size
        plt.subplot(3, 3, 3)
        plt.scatter(df['total_parameters'], df['f1_macro'], alpha=0.6)
        plt.xlabel('Total Parameters')
        plt.ylabel('F1-Macro Score')
        plt.title('Performance vs Model Size')
        
        # 4. Learning Rate vs Performance
        plt.subplot(3, 3, 4)
        df.boxplot(column='f1_macro', by='learning_rate', ax=plt.gca())
        plt.title('F1-Macro vs Learning Rate')
        plt.xlabel('Learning Rate')
        plt.ylabel('F1-Macro Score')
        
        # 5. Hidden Size vs Performance
        plt.subplot(3, 3, 5)
        df.boxplot(column='f1_macro', by='hidden_size', ax=plt.gca())
        plt.title('F1-Macro vs Hidden Size')
        plt.xlabel('Hidden Size')
        plt.ylabel('F1-Macro Score')
        
        # 6. Patch Length vs Performance
        plt.subplot(3, 3, 6)
        df.boxplot(column='f1_macro', by='patch_length', ax=plt.gca())
        plt.title('F1-Macro vs Patch Length')
        plt.xlabel('Patch Length')
        plt.ylabel('F1-Macro Score')
        
        # 7. Embedding Type Comparison
        plt.subplot(3, 3, 7)
        embed_comparison = df.groupby(['case_name', 'embed_type'])['f1_macro'].mean().unstack()
        embed_comparison.plot(kind='bar', ax=plt.gca())
        plt.title('Temporal Embeddings Impact')
        plt.xlabel('Case')
        plt.ylabel('Mean F1-Macro')
        plt.legend(['No Embeddings', 'With Embeddings'])
        plt.xticks(rotation=45)
        
        # 8. Training Efficiency
        plt.subplot(3, 3, 8)
        plt.scatter(df['total_epochs'], df['f1_macro'], alpha=0.6)
        plt.xlabel('Training Epochs')
        plt.ylabel('F1-Macro Score')
        plt.title('Training Efficiency')
        
        # 9. Multi-metric comparison
        plt.subplot(3, 3, 9)
        metrics = ['f1_macro', 'accuracy', 'roc_auc']
        metric_means = df[metrics].mean()
        metric_means.plot(kind='bar', ax=plt.gca())
        plt.title('Overall Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'patchtst_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comprehensive analysis plot saved to {self.output_dir / 'patchtst_comprehensive_analysis.png'}")
    
    def create_case_comparison_plot(self, df: pd.DataFrame):
        """Create detailed case-by-case comparison"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        cases = df['case_name'].unique()
        
        for i, case in enumerate(cases):
            if i < len(axes):
                case_df = df[df['case_name'] == case]
                
                # Create performance distribution
                axes[i].hist(case_df['f1_macro'], bins=15, alpha=0.7, edgecolor='black')
                axes[i].axvline(case_df['f1_macro'].mean(), color='red', linestyle='--', 
                               label=f'Mean: {case_df["f1_macro"].mean():.4f}')
                axes[i].axvline(case_df['f1_macro'].max(), color='green', linestyle='--',
                               label=f'Best: {case_df["f1_macro"].max():.4f}')
                
                axes[i].set_title(f'{case}\n({len(case_df)} experiments)')
                axes[i].set_xlabel('F1-Macro Score')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Remove unused subplot
        if len(cases) < len(axes):
            for j in range(len(cases), len(axes)):
                fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'patchtst_case_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Case comparison plot saved to {self.output_dir / 'patchtst_case_comparison.png'}")
    
    def generate_summary_report(self, df: pd.DataFrame, best_configs: Dict):
        """Generate comprehensive text summary report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
PatchTST Phase 1 Experimental Results Summary
============================================

Generated: {timestamp}
Total Experiments: {len(df):,}
Total Cases: {df['case_name'].nunique()}

OVERALL PERFORMANCE STATISTICS
------------------------------
Mean F1-Macro: {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}
Best F1-Macro: {df['f1_macro'].max():.4f}
Mean Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}
Mean ROC-AUC: {df['roc_auc'].mean():.4f} ± {df['roc_auc'].std():.4f}

BEST CONFIGURATIONS BY CASE
---------------------------
"""
        
        for case_name, config_data in best_configs.items():
            best_config = config_data['best_config']
            stats = config_data['statistics']
            
            report += f"""
{case_name.upper()}:
  Performance:
    Best F1-Macro: {best_config['f1_macro']:.4f}
    Best Accuracy: {best_config['accuracy']:.4f}
    Best ROC-AUC: {best_config['roc_auc']:.4f}
  
  Optimal Hyperparameters:
    Patch Length: {best_config['patch_length']}
    Stride: {best_config['stride']}
    Hidden Size: {best_config['hidden_size']}
    Layers: {best_config['num_hidden_layers']}
    Attention Heads: {best_config['num_attention_heads']}
    Dropout: {best_config['dropout']}
    Learning Rate: {best_config['learning_rate']:.0e}
    Embeddings: {'Yes' if best_config['embed_type'] == 1 else 'No'}
    
  Statistics ({stats['total_experiments']} experiments):
    Mean F1-Macro: {stats['mean_f1_macro']:.4f} ± {stats['std_f1_macro']:.4f}
    Range: {stats['min_f1_macro']:.4f} - {stats['max_f1_macro']:.4f}
"""
        
        # Hyperparameter importance
        importance = self.analyze_hyperparameter_importance(df)
        report += f"""

HYPERPARAMETER IMPORTANCE ANALYSIS
----------------------------------
(Based on correlation with F1-Macro score)

"""
        for param, score in importance.items():
            report += f"  {param}: {score:.4f}\n"
        
        # Model size analysis
        report += f"""

MODEL COMPLEXITY ANALYSIS
-------------------------
Mean Model Parameters: {df['total_parameters'].mean():,.0f}
Parameter Range: {df['total_parameters'].min():,} - {df['total_parameters'].max():,}
Mean Training Epochs: {df['total_epochs'].mean():.1f}

TEMPORAL EMBEDDINGS IMPACT
--------------------------
"""
        
        embed_impact = df.groupby('embed_type')['f1_macro'].agg(['mean', 'std', 'count'])
        report += f"Without Embeddings: {embed_impact.loc[0, 'mean']:.4f} ± {embed_impact.loc[0, 'std']:.4f} ({embed_impact.loc[0, 'count']} experiments)\n"
        if 1 in embed_impact.index:
            report += f"With Embeddings: {embed_impact.loc[1, 'mean']:.4f} ± {embed_impact.loc[1, 'std']:.4f} ({embed_impact.loc[1, 'count']} experiments)\n"
        
        # Save report
        report_file = self.output_dir / f'patchtst_summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Summary report saved to {report_file}")
        
        return report
    
    def run_complete_analysis(self, case_filter: Optional[str] = None):
        """Run complete analysis pipeline"""
        
        print("🔍 Starting PatchTST results analysis...")
        
        if not self.all_results:
            print("❌ No results found to analyze!")
            return
        
        # Create performance summary
        df = self.create_performance_summary()
        
        # Filter by case if specified
        if case_filter:
            df = df[df['case_name'] == case_filter]
            if df.empty:
                print(f"❌ No results found for case: {case_filter}")
                return
        
        print(f"📊 Analyzing {len(df)} experiments across {df['case_name'].nunique()} cases")
        
        # Find best configurations
        best_configs = self.find_best_configurations(df)
        
        # Create visualizations
        print("📊 Creating performance plots...")
        self.create_performance_plots(df)
        self.create_case_comparison_plot(df)
        
        # Generate summary report
        print("📄 Generating summary report...")
        report = self.generate_summary_report(df, best_configs)
        
        # Save processed data
        df.to_csv(self.output_dir / 'patchtst_detailed_results.csv', index=False)
        print(f"💾 Detailed results saved to {self.output_dir / 'patchtst_detailed_results.csv'}")
        
        # Print quick summary to console
        print("\n" + "="*80)
        print("🎉 PATCHTST PHASE 1 ANALYSIS COMPLETE")
        print("="*80)
        print(f"📊 Total experiments analyzed: {len(df):,}")
        print(f"🏆 Best overall F1-Macro: {df['f1_macro'].max():.4f}")
        print(f"📈 Mean F1-Macro: {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}")
        
        print("\n🏆 Best performance by case:")
        for case_name, config_data in best_configs.items():
            best_f1 = config_data['best_config']['f1_macro']
            print(f"  {case_name}: {best_f1:.4f}")
        
        print(f"\n📁 Results saved in: {self.output_dir}")

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Analyze PatchTST experimental results')
    parser.add_argument('--case', type=str, help='Specific case to analyze (optional)')
    parser.add_argument('--output-dir', type=str, help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Set up paths
    results_dir = Path(root) / "results" / "PatchTSTResults"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(root) / "results" / "PatchTSTAnalysis"
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run PatchTST experiments first using RunPatchTSTClassification.py")
        return
    
    # Run analysis
    analyzer = PatchTSTResultsAnalyzer(results_dir, output_dir)
    analyzer.run_complete_analysis(case_filter=args.case)

if __name__ == "__main__":
    main()