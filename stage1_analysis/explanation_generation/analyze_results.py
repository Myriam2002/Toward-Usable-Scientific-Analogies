"""
Explanation Generation Results Visualization and Analysis

This script automatically discovers result files and creates comprehensive visualizations
comparing model performance across different experimental settings.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir path/to/results
    python analyze_results.py --output-dir visualizations
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = 'checkpoints/explanation_generation'
OUTPUT_DIR = 'results/explanation_generation'

SETTING_NAMES = {
    'none': 'Concepts Only',
    'none_description': 'Concepts + Descriptions',
    'unpaired_properties': 'Concepts + Properties',
    'unpaired_properties_description': 'Concepts + Properties + Descriptions',
    'paired_properties': 'Concepts + Paired Mappings',
    'paired_properties_description': 'Concepts + Paired Mappings + Descriptions'
}

# ============================================================================
# DATA LOADING
# ============================================================================

def discover_result_files(results_dir):
    """Discover all ALL_MODELS_combined.csv files in the results directory."""
    pattern = os.path.join(results_dir, '*_ALL_MODELS_combined.csv')
    files = glob.glob(pattern)
    
    result_files = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        setting = filename.replace('_ALL_MODELS_combined.csv', '')
        result_files[setting] = file_path
    
    return result_files


def load_all_results(results_dir):
    """Load all result files and combine into a single DataFrame."""
    result_files = discover_result_files(results_dir)
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    
    print(f"Found {len(result_files)} result files:")
    for setting, path in result_files.items():
        print(f"  • {setting}: {path}")
    
    all_data = []
    for setting, file_path in result_files.items():
        df = pd.read_csv(file_path)
        df['setting'] = setting
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert sbert_similarity to numeric, handling any errors
    combined_df['sbert_similarity'] = pd.to_numeric(combined_df['sbert_similarity'], errors='coerce')
    
    # Add setting display names
    combined_df['setting_name'] = combined_df['setting'].map(SETTING_NAMES)
    
    # Calculate error flag
    combined_df['has_error'] = combined_df['error'].notna() & (combined_df['error'] != '') & (combined_df['error'] != 'None')
    
    print(f"\n✅ Loaded {len(combined_df)} total results")
    print(f"   - Models: {combined_df['model'].nunique()}")
    print(f"   - Settings: {combined_df['setting'].nunique()}")
    print(f"   - Rows: {combined_df['row_index'].nunique()}")
    
    return combined_df


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_heatmap_model_setting_performance(df, output_dir):
    """Create a heatmap showing average SBERT similarity for each model-setting combination."""
    # Calculate average similarity per model-setting
    pivot_data = df.groupby(['model', 'setting_name'])['sbert_similarity'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='model', columns='setting_name', values='sbert_similarity')
    
    # Sort columns by setting order
    setting_order = [SETTING_NAMES[s] for s in SETTING_NAMES.keys() if SETTING_NAMES[s] in pivot_table.columns]
    pivot_table = pivot_table[setting_order]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, center=0.75, cbar_kws={'label': 'SBERT Similarity'})
    plt.title('Model Performance Across Experimental Settings\n(Average SBERT Similarity)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Experimental Setting', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_heatmap_model_setting_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: 1_heatmap_model_setting_performance.png")


def plot_model_comparison_boxplot(df, output_dir):
    """Create box plots comparing model performance across all settings."""
    plt.figure(figsize=(16, 8))
    
    # Sort models by median performance
    model_order = df.groupby('model')['sbert_similarity'].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df, x='model', y='sbert_similarity', order=model_order, palette='Set3')
    plt.axhline(y=0.75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target: 0.75')
    plt.axhline(y=df['sbert_similarity'].mean(), color='blue', linestyle='--', linewidth=1, alpha=0.5, 
                label=f'Overall Mean: {df["sbert_similarity"].mean():.3f}')
    
    plt.title('Model Performance Distribution Across All Settings', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('SBERT Similarity Score', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_model_comparison_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: 2_model_comparison_boxplot.png")


def plot_setting_comparison_boxplot(df, output_dir):
    """Create box plots comparing performance across experimental settings."""
    plt.figure(figsize=(14, 8))
    
    # Order by mean performance
    setting_order = df.groupby('setting_name')['sbert_similarity'].mean().sort_values(ascending=False).index
    
    sns.boxplot(data=df, x='setting_name', y='sbert_similarity', order=setting_order, palette='Set2')
    plt.axhline(y=0.75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target: 0.75')
    
    plt.title('Performance Distribution Across Experimental Settings', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Experimental Setting', fontsize=12, fontweight='bold')
    plt.ylabel('SBERT Similarity Score', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_setting_comparison_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: 3_setting_comparison_boxplot.png")


def plot_model_ranking_by_setting(df, output_dir):
    """Create a bar chart showing model rankings for each setting."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    settings = [s for s in SETTING_NAMES.keys() if SETTING_NAMES[s] in df['setting_name'].unique()]
    
    for idx, setting in enumerate(settings):
        if idx >= len(axes):
            break
            
        setting_name = SETTING_NAMES[setting]
        setting_df = df[df['setting_name'] == setting_name]
        
        # Calculate mean performance per model
        model_perf = setting_df.groupby('model')['sbert_similarity'].mean().sort_values(ascending=False)
        
        ax = axes[idx]
        colors = plt.cm.RdYlGn(model_perf.values)
        ax.barh(range(len(model_perf)), model_perf.values, color=colors)
        ax.set_yticks(range(len(model_perf)))
        ax.set_yticklabels(model_perf.index, fontsize=9)
        ax.set_xlabel('SBERT Similarity', fontsize=10)
        ax.set_title(setting_name, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.axvline(x=0.75, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(model_perf.values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.suptitle('Model Rankings by Experimental Setting', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_model_ranking_by_setting.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: 4_model_ranking_by_setting.png")


def plot_error_rate_analysis(df, output_dir):
    """Analyze and visualize error rates across models and settings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Error rate by model
    error_by_model = df.groupby('model')['has_error'].mean() * 100
    error_by_model = error_by_model.sort_values(ascending=False)
    
    colors1 = ['red' if x > 5 else 'orange' if x > 1 else 'green' for x in error_by_model.values]
    ax1.barh(range(len(error_by_model)), error_by_model.values, color=colors1)
    ax1.set_yticks(range(len(error_by_model)))
    ax1.set_yticklabels(error_by_model.index)
    ax1.set_xlabel('Error Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Error Rate by Model', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(error_by_model.values):
        ax1.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # Error rate by setting
    error_by_setting = df.groupby('setting_name')['has_error'].mean() * 100
    error_by_setting = error_by_setting.sort_values(ascending=False)
    
    colors2 = ['red' if x > 5 else 'orange' if x > 1 else 'green' for x in error_by_setting.values]
    ax2.bar(range(len(error_by_setting)), error_by_setting.values, color=colors2)
    ax2.set_xticks(range(len(error_by_setting)))
    ax2.set_xticklabels(error_by_setting.index, rotation=45, ha='right')
    ax2.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Error Rate by Setting', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(error_by_setting.values):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    
    plt.suptitle('Error Rate Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_error_rate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: 5_error_rate_analysis.png")


def plot_top_bottom_models(df, output_dir):
    """Compare top 3 and bottom 3 models across settings."""
    # Get overall model performance
    model_performance = df.groupby('model')['sbert_similarity'].mean().sort_values()
    top_3 = model_performance.tail(3).index.tolist()
    bottom_3 = model_performance.head(3).index.tolist()
    
    # Filter data
    filtered_df = df[df['model'].isin(top_3 + bottom_3)].copy()
    filtered_df['group'] = filtered_df['model'].apply(lambda x: 'Top 3' if x in top_3 else 'Bottom 3')
    
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar plot
    setting_order = [SETTING_NAMES[s] for s in SETTING_NAMES.keys() if SETTING_NAMES[s] in filtered_df['setting_name'].unique()]
    
    # Calculate means
    plot_data = filtered_df.groupby(['setting_name', 'model', 'group'])['sbert_similarity'].mean().reset_index()
    
    x = np.arange(len(setting_order))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors_top = ['#2ecc71', '#27ae60', '#229954']
    colors_bottom = ['#e74c3c', '#c0392b', '#a93226']
    
    for i, model in enumerate(top_3):
        model_data = plot_data[(plot_data['model'] == model) & (plot_data['setting_name'].isin(setting_order))]
        model_data = model_data.set_index('setting_name').reindex(setting_order, fill_value=0)
        ax.bar(x + i * width, model_data['sbert_similarity'], width, label=model, color=colors_top[i])
    
    for i, model in enumerate(bottom_3):
        model_data = plot_data[(plot_data['model'] == model) & (plot_data['setting_name'].isin(setting_order))]
        model_data = model_data.set_index('setting_name').reindex(setting_order, fill_value=0)
        ax.bar(x + (i + 3) * width, model_data['sbert_similarity'], width, label=model, color=colors_bottom[i])
    
    ax.set_xlabel('Experimental Setting', fontsize=12, fontweight='bold')
    ax.set_ylabel('SBERT Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 3 vs Bottom 3 Models Performance Across Settings', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(setting_order, rotation=45, ha='right')
    ax.legend(ncol=2, loc='upper left', fontsize=9)
    ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Target: 0.75')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_top_bottom_models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: 6_top_bottom_models_comparison.png")


def plot_setting_impact_on_models(df, output_dir):
    """Show how each setting impacts different models (improvement/degradation)."""
    # Use 'none' as baseline
    baseline_setting = 'Concepts Only'
    
    if baseline_setting not in df['setting_name'].unique():
        print("⚠️  Baseline setting 'none' not found, skipping setting impact analysis")
        return
    
    baseline = df[df['setting_name'] == baseline_setting].groupby('model')['sbert_similarity'].mean()
    
    improvements = {}
    for setting in df['setting_name'].unique():
        if setting == baseline_setting:
            continue
        setting_perf = df[df['setting_name'] == setting].groupby('model')['sbert_similarity'].mean()
        improvements[setting] = setting_perf - baseline
    
    if not improvements:
        print("⚠️  No other settings found for comparison")
        return
    
    # Create DataFrame
    improvement_df = pd.DataFrame(improvements)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(improvement_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0, cbar_kws={'label': 'Change in SBERT Similarity'})
    plt.title(f'Performance Change Relative to Baseline ({baseline_setting})\nPositive = Improvement, Negative = Degradation', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Experimental Setting', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_setting_impact_on_models.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: 7_setting_impact_on_models.png")


def plot_consistency_analysis(df, output_dir):
    """Analyze consistency (variance) of model performance."""
    # Calculate standard deviation for each model across all runs
    consistency = df.groupby('model')['sbert_similarity'].agg(['mean', 'std']).sort_values('std')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn_r(consistency['std'].values / consistency['std'].max())
    ax.barh(range(len(consistency)), consistency['std'], color=colors)
    ax.set_yticks(range(len(consistency)))
    ax.set_yticklabels(consistency.index)
    ax.set_xlabel('Standard Deviation (Lower = More Consistent)', fontsize=11, fontweight='bold')
    ax.set_title('Model Consistency Analysis\n(Lower std = More Consistent Performance)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (std_val, mean_val) in enumerate(zip(consistency['std'].values, consistency['mean'].values)):
        ax.text(std_val + 0.002, i, f'{std_val:.3f}\n(μ={mean_val:.3f})', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_consistency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: 8_consistency_analysis.png")


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def generate_summary_statistics(df, output_dir):
    """Generate comprehensive summary statistics."""
    summary = {}
    
    # Overall statistics
    summary['overall'] = {
        'total_experiments': len(df),
        'total_models': df['model'].nunique(),
        'total_settings': df['setting'].nunique(),
        'mean_sbert_similarity': float(df['sbert_similarity'].mean()),
        'median_sbert_similarity': float(df['sbert_similarity'].median()),
        'std_sbert_similarity': float(df['sbert_similarity'].std()),
        'error_rate': float(df['has_error'].mean() * 100)
    }
    
    # Model statistics
    model_stats = df.groupby('model')['sbert_similarity'].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
    model_errors = df.groupby('model')['has_error'].mean() * 100
    model_stats['error_rate'] = model_errors
    model_stats = model_stats.sort_values('mean', ascending=False)
    summary['model_rankings'] = model_stats.to_dict('index')
    
    # Setting statistics
    setting_stats = df.groupby('setting_name')['sbert_similarity'].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
    setting_errors = df.groupby('setting_name')['has_error'].mean() * 100
    setting_stats['error_rate'] = setting_errors
    setting_stats = setting_stats.sort_values('mean', ascending=False)
    summary['setting_rankings'] = setting_stats.to_dict('index')
    
    # Best combinations
    best_combos = df.groupby(['model', 'setting_name'])['sbert_similarity'].mean().sort_values(ascending=False).head(10)
    summary['top_10_combinations'] = {f"{model} - {setting}": float(score) 
                                       for (model, setting), score in best_combos.items()}
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'summary_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Created: summary_statistics.json")
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'model_rankings.csv')
    model_stats.to_csv(csv_path)
    print(f"✅ Created: model_rankings.csv")
    
    csv_path = os.path.join(output_dir, 'setting_rankings.csv')
    setting_stats.to_csv(csv_path)
    print(f"✅ Created: setting_rankings.csv")
    
    return summary


def print_summary_report(summary):
    """Print a formatted summary report to console."""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print("\n📊 OVERALL STATISTICS")
    print(f"  • Total Experiments: {summary['overall']['total_experiments']}")
    print(f"  • Models Tested: {summary['overall']['total_models']}")
    print(f"  • Settings Tested: {summary['overall']['total_settings']}")
    print(f"  • Mean SBERT Similarity: {summary['overall']['mean_sbert_similarity']:.4f}")
    print(f"  • Median SBERT Similarity: {summary['overall']['median_sbert_similarity']:.4f}")
    print(f"  • Std Dev: {summary['overall']['std_sbert_similarity']:.4f}")
    print(f"  • Error Rate: {summary['overall']['error_rate']:.2f}%")
    
    print("\n🏆 TOP 5 MODELS (by mean performance)")
    for i, (model, stats) in enumerate(list(summary['model_rankings'].items())[:5], 1):
        print(f"  {i}. {model}: {stats['mean']:.4f} (±{stats['std']:.4f})")
    
    print("\n📈 TOP 5 SETTINGS (by mean performance)")
    for i, (setting, stats) in enumerate(list(summary['setting_rankings'].items())[:5], 1):
        print(f"  {i}. {setting}: {stats['mean']:.4f} (±{stats['std']:.4f})")
    
    print("\n⭐ TOP 5 MODEL-SETTING COMBINATIONS")
    for i, (combo, score) in enumerate(list(summary['top_10_combinations'].items())[:5], 1):
        print(f"  {i}. {combo}: {score:.4f}")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize explanation generation experiment results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='checkpoints/explanation_generation',
        help='Directory containing result files (default: checkpoints/explanation_generation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/explanation_generation',
        help='Output directory for visualizations (default: results/explanation_generation)'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    print("="*80)
    print("EXPLANATION GENERATION RESULTS ANALYSIS")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n✅ Output directory: {args.output_dir}")
    
    # Load data
    print(f"\n📂 Loading results from: {args.results_dir}")
    df = load_all_results(args.results_dir)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    plot_heatmap_model_setting_performance(df, args.output_dir)
    plot_model_comparison_boxplot(df, args.output_dir)
    plot_setting_comparison_boxplot(df, args.output_dir)
    plot_model_ranking_by_setting(df, args.output_dir)
    plot_error_rate_analysis(df, args.output_dir)
    plot_top_bottom_models(df, args.output_dir)
    plot_setting_impact_on_models(df, args.output_dir)
    plot_consistency_analysis(df, args.output_dir)
    
    # Generate summary statistics
    print("\n📊 Generating summary statistics...")
    summary = generate_summary_statistics(df, args.output_dir)
    
    # Print report
    print_summary_report(summary)
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")
    print(f"   Generated {len(glob.glob(os.path.join(args.output_dir, '*.png')))} visualizations")


if __name__ == "__main__":
    main()

