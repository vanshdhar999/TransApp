import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_all_results():
    """Load all TST experimental results"""
    root = Path.cwd().parent
    results_dir = root / 'results' / 'TransAppResults_TST'
    
    all_results = []
    
    for embed_dir in ['None', 'Embed']:
        embed_path = results_dir / embed_dir
        if embed_path.exists():
            for json_file in embed_path.rglob('*_results.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        data['embed_type_name'] = embed_dir
                        all_results.append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    
    return all_results

def create_performance_comparison():
    """Create comprehensive performance comparison"""
    results = load_all_results()
    
    if not results:
        print("No results found!")
        return
    
    # Extract metrics
    data = []
    for result in results:
        if 'results' in result and 'quantile_metrics' in result['results']:
            metrics = result['results']['quantile_metrics']
            config = result.get('configuration', {})
            
            data.append({
                'case_name': result.get('case_name', 'unknown'),
                'model_name': result.get('model_name', 'unknown'),
                'embed_type': result.get('embed_type_name', 'unknown'),
                'norm_type': config.get('norm_type', 'unknown'),
                'dim_model': result.get('dim_model', 0),
                'f1_score': metrics.get('F1_SCORE', 0),
                'accuracy': metrics.get('ACCURACY', 0),
                'roc_auc': metrics.get('ROC_AUC_SCORE', 0),
                'precision': metrics.get('PRECISION', 0),
                'recall': metrics.get('RECALL', 0)
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No valid data extracted!")
        return
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Performance by embedding type
    embed_perf = df.groupby('embed_type')[['f1_score', 'accuracy', 'roc_auc']].mean()
    embed_perf.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Performance by Embedding Type')
    axes[0,0].legend()
    
    # 2. Performance by case
    case_perf = df.groupby('case_name')['f1_score'].mean().sort_values()
    case_perf.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('F1-Score by Case')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Normalization comparison
    if len(df['norm_type'].unique()) > 1:
        norm_perf = df.groupby('norm_type')[['f1_score', 'accuracy']].mean()
        norm_perf.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Performance by Normalization Type')
    
    # 4. Dimension comparison
    if len(df['dim_model'].unique()) > 1:
        dim_perf = df.groupby('dim_model')['f1_score'].mean()
        dim_perf.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('F1-Score by Model Dimension')
    
    plt.tight_layout()
    plt.savefig('tst_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print("📊 STATISTICAL ANALYSIS")
    print("="*50)
    
    # Compare embedding types
    if len(df['embed_type'].unique()) == 2:
        none_scores = df[df['embed_type'] == 'None']['f1_score']
        embed_scores = df[df['embed_type'] == 'Embed']['f1_score']
        
        if len(none_scores) > 0 and len(embed_scores) > 0:
            t_stat, p_value = stats.ttest_ind(none_scores, embed_scores)
            print(f"Embedding vs None comparison:")
            print(f"  None mean: {none_scores.mean():.4f} ± {none_scores.std():.4f}")
            print(f"  Embed mean: {embed_scores.mean():.4f} ± {embed_scores.std():.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    return df

if __name__ == "__main__":
    print("🔍 Analyzing TST experimental results...")
    df = create_performance_comparison()
    
    if df is not None and not df.empty:
        print(f"\n📈 Summary: Analyzed {len(df)} experiments")
        print(f"Cases: {', '.join(df['case_name'].unique())}")
        print(f"Embedding types: {', '.join(df['embed_type'].unique())}")
        print(f"Models: {', '.join(df['model_name'].unique())}")
