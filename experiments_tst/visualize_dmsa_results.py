#!/usr/bin/env python3
"""
DMSA vs Standard TST Attention Visualization
===========================================

This script creates comprehensive visualizations comparing DMSA (Diagonally Masked Self-Attention)
with Standard Self-Attention in TST architecture for appliance detection.

Key Comparisons:
1. F1-Macro performance across attention mechanisms
2. Impact of temporal embeddings 
3. Statistical significance analysis
4. Confusion matrix heatmaps
5. Performance stability across random seeds

Author: Enhanced TransApp Framework  
Date: October 2025
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_experiment_results(results_file):
    """Load DMSA-TST experiment results"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Loaded experiment: {data['experiment_info']['case_name']}")
    print(f"📅 Timestamp: {data['timestamp']}")
    print(f"🎯 Total experiments: {len(data['all_results'])}")
    
    return data

def parse_results_to_dataframe(data):
    """Convert results to pandas DataFrame for analysis"""
    results_list = []
    
    for result in data['all_results']:
        row = {
            'attention_type': result['attention_type'],
            'mask_diag': result['mask_diag'],
            'embed_type': 'With Embeddings' if result['embed_type'] == 1 else 'No Embeddings',
            'embed_type_num': result['embed_type'],
            'random_seed': result['random_seed'],
            **result['results']  # Unpack all metrics
        }
        results_list.append(row)
    
    df = pd.DataFrame(results_list)
    
    # Create combined category for better visualization
    df['config'] = df['attention_type'] + ' + ' + df['embed_type']
    
    return df

def create_f1_macro_comparison_plot(df, save_path):
    """Create F1-Macro comparison plot with error bars"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Box plot showing distribution across seeds
    sns.boxplot(data=df, x='attention_type', y='F1_SCORE_MACRO', 
                hue='embed_type', ax=ax1, width=0.6)
    ax1.set_title('F1-Macro Score Distribution\nDMSA vs Standard Attention', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Attention Mechanism', fontsize=12)
    ax1.set_ylabel('F1-Macro Score', fontsize=12)
    ax1.legend(title='Embedding Type', loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add mean values as text
    for i, attention in enumerate(['DMSA', 'Standard']):
        for j, embed in enumerate(['No Embeddings', 'With Embeddings']):
            subset = df[(df['attention_type'] == attention) & (df['embed_type'] == embed)]
            mean_val = subset['F1_SCORE_MACRO'].mean()
            x_pos = i + (j - 0.5) * 0.4
            ax1.text(x_pos, mean_val + 0.01, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Bar plot with error bars
    grouped_stats = df.groupby(['attention_type', 'embed_type'])['F1_SCORE_MACRO'].agg(['mean', 'std']).reset_index()
    
    x_pos = np.arange(len(grouped_stats))
    bars = ax2.bar(x_pos, grouped_stats['mean'], yerr=grouped_stats['std'], 
                   capsize=5, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    ax2.set_title('F1-Macro Performance\nMean ± Standard Deviation', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Macro Score', fontsize=12)
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{row['attention_type']}\n{row['embed_type']}" 
                        for _, row in grouped_stats.iterrows()], fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, grouped_stats['mean'], grouped_stats['std'])):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 0.005,
                f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path / 'f1_macro_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return grouped_stats

def create_comprehensive_metrics_heatmap(df, save_path):
    """Create heatmap of all metrics for comprehensive comparison"""
    
    # Select key metrics for heatmap
    metrics = ['F1_SCORE_MACRO', 'ACCURACY', 'PRECISION_MACRO', 'RECALL_MACRO', 'ROC_AUC_SCORE']
    
    # Calculate mean values for each configuration
    heatmap_data = df.groupby(['attention_type', 'embed_type'])[metrics].mean()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Transpose for better layout (metrics as columns)
    heatmap_data_transposed = heatmap_data.T
    
    sns.heatmap(heatmap_data_transposed, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0.6, ax=ax, cbar_kws={'label': 'Score'})
    
    ax.set_title('Performance Metrics Heatmap\nDMSA vs Standard Attention', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Metrics', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_stability_analysis_plot(df, save_path):
    """Analyze performance stability across random seeds"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    configs = df['config'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, config in enumerate(configs):
        ax = axes[i//2, i%2]
        config_data = df[df['config'] == config].sort_values('random_seed')
        
        # Plot F1-Macro across seeds
        ax.plot(config_data['random_seed'], config_data['F1_SCORE_MACRO'], 
               'o-', color=colors[i], linewidth=2, markersize=8, label=config)
        ax.fill_between(config_data['random_seed'], 
                       config_data['F1_SCORE_MACRO'] - 0.01,
                       config_data['F1_SCORE_MACRO'] + 0.01,
                       alpha=0.2, color=colors[i])
        
        ax.set_title(f'Stability: {config}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Random Seed', fontsize=10)
        ax.set_ylabel('F1-Macro Score', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.6, 0.65])
        
        # Add mean line
        mean_f1 = config_data['F1_SCORE_MACRO'].mean()
        ax.axhline(y=mean_f1, color=colors[i], linestyle='--', alpha=0.7)
        ax.text(0.5, mean_f1 + 0.005, f'Mean: {mean_f1:.3f}', 
               transform=ax.get_xaxis_transform(), ha='center', fontweight='bold')
        
        # Add standard deviation
        std_f1 = config_data['F1_SCORE_MACRO'].std()
        ax.text(0.98, 0.95, f'Std: {std_f1:.4f}', transform=ax.transAxes, 
               ha='right', va='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Performance Stability Across Random Seeds\nLower Standard Deviation = More Stable', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_attention_mechanism_comparison(df, save_path):
    """Direct comparison between DMSA and Standard attention"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Radar chart comparing key metrics
    metrics = ['F1_SCORE_MACRO', 'ACCURACY', 'PRECISION_MACRO', 'RECALL_MACRO', 'ROC_AUC_SCORE']
    
    # Calculate means for each attention type (averaged across embeddings and seeds)
    dmsa_means = df[df['attention_type'] == 'DMSA'][metrics].mean()
    standard_means = df[df['attention_type'] == 'Standard'][metrics].mean()
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    dmsa_values = dmsa_means.tolist()
    standard_values = standard_means.tolist()
    dmsa_values += dmsa_values[:1]
    standard_values += standard_values[:1]
    
    ax1.plot(angles, dmsa_values, 'o-', linewidth=2, label='DMSA', color='#d62728')
    ax1.fill(angles, dmsa_values, alpha=0.25, color='#d62728')
    ax1.plot(angles, standard_values, 'o-', linewidth=2, label='Standard', color='#1f77b4')
    ax1.fill(angles, standard_values, alpha=0.25, color='#1f77b4')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(['F1-Macro', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC'])
    ax1.set_ylim(0.6, 0.85)
    ax1.set_title('Overall Performance Comparison\nDMSA vs Standard Attention', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Win/Loss comparison
    metrics_comparison = []
    for metric in metrics:
        dmsa_mean = df[df['attention_type'] == 'DMSA'][metric].mean()
        standard_mean = df[df['attention_type'] == 'Standard'][metric].mean()
        winner = 'DMSA' if dmsa_mean > standard_mean else 'Standard'
        difference = abs(dmsa_mean - standard_mean)
        metrics_comparison.append({
            'metric': metric.replace('_', ' ').title(),
            'winner': winner,
            'dmsa_score': dmsa_mean,
            'standard_score': standard_mean,
            'difference': difference
        })
    
    comparison_df = pd.DataFrame(metrics_comparison)
    
    # Create bar chart showing differences
    colors = ['#d62728' if winner == 'DMSA' else '#1f77b4' for winner in comparison_df['winner']]
    bars = ax2.bar(range(len(comparison_df)), comparison_df['difference'], 
                   color=colors, alpha=0.7)
    
    ax2.set_title('Performance Difference by Metric\n(Positive = Winner Advantage)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Metrics', fontsize=12)
    ax2.set_ylabel('Absolute Difference', fontsize=12)
    ax2.set_xticks(range(len(comparison_df)))
    ax2.set_xticklabels([m.replace(' ', '\n') for m in comparison_df['metric']], fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add winner labels
    for i, (bar, winner, diff) in enumerate(zip(bars, comparison_df['winner'], comparison_df['difference'])):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{winner}\n+{diff:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path / 'attention_mechanism_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def create_embedding_impact_analysis(df, save_path):
    """Analyze the impact of temporal embeddings on both attention mechanisms"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate embedding impact for each attention type
    impact_data = []
    for attention in ['DMSA', 'Standard']:
        no_embed = df[(df['attention_type'] == attention) & (df['embed_type'] == 'No Embeddings')]['F1_SCORE_MACRO'].mean()
        with_embed = df[(df['attention_type'] == attention) & (df['embed_type'] == 'With Embeddings')]['F1_SCORE_MACRO'].mean()
        impact = with_embed - no_embed
        
        impact_data.append({
            'attention_type': attention,
            'no_embeddings': no_embed,
            'with_embeddings': with_embed,
            'impact': impact,
            'impact_percent': (impact / no_embed) * 100
        })
    
    impact_df = pd.DataFrame(impact_data)
    
    # Plot 1: Before/After embedding comparison
    x = np.arange(len(impact_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, impact_df['no_embeddings'], width, 
                   label='No Embeddings', alpha=0.8, color='#ff7f0e')
    bars2 = ax1.bar(x + width/2, impact_df['with_embeddings'], width,
                   label='With Embeddings', alpha=0.8, color='#2ca02c')
    
    ax1.set_title('Impact of Temporal Embeddings\nF1-Macro Score Comparison', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1-Macro Score', fontsize=12)
    ax1.set_xlabel('Attention Mechanism', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(impact_df['attention_type'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Impact magnitude
    colors = ['#d62728' if impact < 0 else '#2ca02c' for impact in impact_df['impact']]
    bars = ax2.bar(impact_df['attention_type'], impact_df['impact'], 
                   color=colors, alpha=0.7)
    
    ax2.set_title('Embedding Impact Magnitude\n(Negative = Embeddings Hurt Performance)', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Macro Change', fontsize=12)
    ax2.set_xlabel('Attention Mechanism', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add impact labels
    for bar, impact, percent in zip(bars, impact_df['impact'], impact_df['impact_percent']):
        ax2.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + (0.002 if impact >= 0 else -0.004),
                f'{impact:+.3f}\n({percent:+.1f}%)', 
                ha='center', va='bottom' if impact >= 0 else 'top', 
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / 'embedding_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return impact_df

def create_summary_report(df, grouped_stats, comparison_df, impact_df, save_path):
    """Create a text summary report"""
    
    report_lines = [
        "=" * 80,
        "DMSA vs STANDARD TST ATTENTION MECHANISM ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: CER Cooker Case Detection",
        f"Total Experiments: {len(df)} (4 configs × 3 seeds)",
        "",
        "EXECUTIVE SUMMARY:",
        "-" * 40,
    ]
    
    # Find best overall configuration
    best_config = grouped_stats.loc[grouped_stats['mean'].idxmax()]
    report_lines.extend([
        f"🏆 Best Configuration: {best_config['attention_type']} + {best_config['embed_type']}",
        f"   F1-Macro: {best_config['mean']:.4f} ± {best_config['std']:.4f}",
        "",
    ])
    
    # DMSA vs Standard comparison
    dmsa_mean = df[df['attention_type'] == 'DMSA']['F1_SCORE_MACRO'].mean()
    standard_mean = df[df['attention_type'] == 'Standard']['F1_SCORE_MACRO'].mean()
    
    if dmsa_mean > standard_mean:
        winner = "DMSA"
        advantage = dmsa_mean - standard_mean
    else:
        winner = "Standard"
        advantage = standard_mean - dmsa_mean
    
    report_lines.extend([
        "ATTENTION MECHANISM COMPARISON:",
        "-" * 40,
        f"DMSA Average F1-Macro:     {dmsa_mean:.4f}",
        f"Standard Average F1-Macro: {standard_mean:.4f}",
        f"Winner: {winner} (+{advantage:.4f} advantage)",
        "",
    ])
    
    # Embedding analysis
    report_lines.extend([
        "TEMPORAL EMBEDDING IMPACT:",
        "-" * 40,
    ])
    
    for _, row in impact_df.iterrows():
        direction = "improves" if row['impact'] > 0 else "hurts"
        report_lines.append(
            f"{row['attention_type']:8s}: Embeddings {direction} by {row['impact']:+.4f} ({row['impact_percent']:+.1f}%)"
        )
    
    report_lines.extend([
        "",
        "STABILITY ANALYSIS:",
        "-" * 40,
    ])
    
    # Calculate stability (lower std = more stable)
    for config in grouped_stats['attention_type'].unique():
        for embed in grouped_stats['embed_type'].unique():
            subset = grouped_stats[
                (grouped_stats['attention_type'] == config) & 
                (grouped_stats['embed_type'] == embed)
            ]
            if not subset.empty:
                std_val = subset['std'].iloc[0]
                stability = "High" if std_val < 0.01 else "Medium" if std_val < 0.02 else "Low"
                report_lines.append(
                    f"{config} + {embed:15s}: {stability:6s} stability (σ={std_val:.4f})"
                )
    
    # Key findings
    report_lines.extend([
        "",
        "KEY FINDINGS:",
        "-" * 40,
        "1. " + ("DMSA shows superior performance" if winner == "DMSA" else "Standard attention performs better"),
        f"2. Temporal embeddings generally {'help' if impact_df['impact'].mean() > 0 else 'hurt'} both mechanisms",
        f"3. Performance is {'consistent' if grouped_stats['std'].mean() < 0.02 else 'variable'} across random seeds",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        f"- Use {best_config['attention_type']} attention mechanism",
        f"- {'Include' if 'With' in best_config['embed_type'] else 'Exclude'} temporal embeddings",
        f"- Expected F1-Macro performance: ~{best_config['mean']:.3f}",
        "",
        "=" * 80
    ])
    
    # Save report
    with open(save_path / 'analysis_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print to console
    for line in report_lines:
        print(line)

def main():
    """Main visualization pipeline"""
    
    # Setup paths
    results_file = Path("/home/user/vansh/ISP/TransApp/results/TransAppResults_TST/dmsa_tst_comparison_cooker_case_96_BatchNorm_5ep_20251016_212629.json")
    save_path = Path("/home/user/vansh/ISP/TransApp/results/TransAppResults_TST/dmsa_analysis_plots")
    save_path.mkdir(exist_ok=True)
    
    print("🎨 Starting DMSA vs Standard TST Visualization Pipeline")
    print("=" * 60)
    
    # Load and parse data
    data = load_experiment_results(results_file)
    df = parse_results_to_dataframe(data)
    
    print(f"\n📊 Parsed {len(df)} experimental results")
    print(f"Configurations tested: {df['config'].nunique()}")
    
    # Create visualizations
    print("\n🎯 Creating visualizations...")
    
    # 1. F1-Macro comparison
    print("  1. F1-Macro performance comparison...")
    grouped_stats = create_f1_macro_comparison_plot(df, save_path)
    
    # 2. Comprehensive metrics heatmap
    print("  2. Comprehensive metrics heatmap...")
    create_comprehensive_metrics_heatmap(df, save_path)
    
    # 3. Stability analysis
    print("  3. Performance stability analysis...")
    create_stability_analysis_plot(df, save_path)
    
    # 4. Attention mechanism comparison
    print("  4. Attention mechanism direct comparison...")
    comparison_df = create_attention_mechanism_comparison(df, save_path)
    
    # 5. Embedding impact analysis
    print("  5. Temporal embedding impact analysis...")
    impact_df = create_embedding_impact_analysis(df, save_path)
    
    # 6. Generate summary report
    print("  6. Generating summary report...")
    create_summary_report(df, grouped_stats, comparison_df, impact_df, save_path)
    
    print(f"\n✅ All visualizations saved to: {save_path}")
    print("📈 Analysis complete!")

if __name__ == "__main__":
    main()