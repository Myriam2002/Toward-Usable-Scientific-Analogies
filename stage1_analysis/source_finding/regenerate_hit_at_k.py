"""
Regenerate plots with custom labels
"""

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style to match original
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.dpi'] = 100

# Color palette - 4 distinct colors that work well together
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']  # Blue, Orange, Green, Red

# Label mapping: old -> new
LABEL_MAPPING = {
    'name_properties': 'name_subconcepts',
    'name_properties_background': 'name_subconcepts_background'
}


def categorize_performance(gold_rank: int) -> str:
    """Categorize performance based on gold rank"""
    if gold_rank == -1:
        return 'Not Found'
    elif gold_rank == 1:
        return 'Perfect (Rank 1)'
    elif gold_rank <= 3:
        return 'Excellent (Rank 2-3)'
    elif gold_rank <= 5:
        return 'Good (Rank 4-5)'
    elif gold_rank <= 10:
        return 'Fair (Rank 6-10)'
    else:
        return 'Poor (Rank >10)'


def load_raw_data(results_dir: Path) -> dict:
    """Load raw RAG result CSV files"""
    data = {}
    rag_files = list(results_dir.glob("rag_results_*.csv"))
    
    for file_path in rag_files:
        # Extract embedding mode from filename
        mode = file_path.stem.replace("rag_results_", "")
        # Apply label mapping
        mode = LABEL_MAPPING.get(mode, mode)
        
        print(f"  Loading {file_path.name} as '{mode}'...")
        df = pd.read_csv(file_path)
        data[mode] = df
    
    return data


def plot_hit_at_k_curves(metrics_df, output_dir):
    """Plot Hit@K curves"""
    hit_columns = [col for col in metrics_df.columns if col.startswith('Hit@')]
    
    if not hit_columns:
        print("No Hit@K metrics available")
        return
    
    k_values = sorted([int(col.split('@')[1]) for col in hit_columns])
    
    plt.figure(figsize=(10, 6))
    
    for idx, mode in enumerate(metrics_df.index):
        hit_scores = [metrics_df.loc[mode, f'Hit@{k}'] for k in k_values]
        color = COLORS[idx % len(COLORS)]
        plt.plot(k_values, hit_scores, marker='o', label=mode, linewidth=2, color=color)
    
    plt.xlabel('K (Rank Threshold)', fontsize=12)
    plt.ylabel('Hit@K (Success Rate)', fontsize=12)
    plt.title('Hit@K: Success Rate at Different Rank Thresholds', fontsize=14, fontweight='bold')
    plt.legend(title='Embedding Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "02_hit_at_k_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rank_distribution(data_dict, output_dir):
    """Plot histogram of gold ranks by mode"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    modes = list(data_dict.keys())[:4]
    
    for idx, mode in enumerate(modes):
        df = data_dict[mode]
        
        # Filter out -1 (not found) for histogram
        found_ranks = df[df['gold_rank'] > 0]['gold_rank']
        
        color = COLORS[idx % len(COLORS)]
        
        if len(found_ranks) > 0:
            # Use integer bins for rank data
            max_rank = int(found_ranks.max())
            min_rank = int(found_ranks.min())
            bins = np.arange(min_rank - 0.5, max_rank + 1.5, 1)  # Center bins on integers
            
            axes[idx].hist(found_ranks, bins=bins, edgecolor='black', alpha=0.7, color=color)
            axes[idx].axvline(found_ranks.median(), color='black', linestyle='--', 
                             linewidth=2, label=f'Median: {found_ranks.median():.0f}')
            axes[idx].set_xlabel('Gold Rank', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{mode}\n(Found: {len(found_ranks)}/{len(df)})', fontsize=14, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Set integer x-axis ticks
            axes[idx].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            axes[idx].set_title(f'{mode}', fontsize=14, fontweight='bold')
    
    plt.suptitle('Distribution of Gold Ranks by Embedding Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "03_rank_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_failure_analysis(data_dict, output_dir):
    """Visualize failure categories and frequencies"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    modes = list(data_dict.keys())[:4]
    
    for idx, mode in enumerate(modes):
        df = data_dict[mode].copy()
        
        # Categorize performance
        df['category'] = df['gold_rank'].apply(categorize_performance)
        
        category_counts = df['category'].value_counts()
        
        # Sort categories in logical order
        category_order = ['Perfect (Rank 1)', 'Excellent (Rank 2-3)', 'Good (Rank 4-5)', 
                        'Fair (Rank 6-10)', 'Poor (Rank >10)', 'Not Found']
        category_counts = category_counts.reindex([c for c in category_order if c in category_counts.index], 
                                                 fill_value=0)
        
        # Use gradient colors from good to bad performance
        num_cats = len(category_counts)
        if num_cats >= 6:
            colors_list = ['#55A868', '#8FBC8F', '#B0C4DE', '#F4A460', '#E9967A', '#C44E52']
        elif num_cats >= 4:
            colors_list = ['#55A868', '#B0C4DE', '#F4A460', '#C44E52']
        else:
            colors_list = COLORS[:num_cats]
        colors_list = colors_list[:num_cats]
        
        axes[idx].bar(range(len(category_counts)), category_counts.values, 
                     color=colors_list, alpha=0.7, edgecolor='black')
        axes[idx].set_xticks(range(len(category_counts)))
        axes[idx].set_xticklabels(category_counts.index, rotation=45, ha='right', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=12)
        axes[idx].set_title(f'{mode}', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Performance Categories by Embedding Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "07_failure_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_dir = Path("results")
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data for rank distribution and failure analysis
    print("Loading raw data...")
    data_dict = load_raw_data(results_dir)
    print(f"Loaded data for modes: {list(data_dict.keys())}")
    
    # Load the comprehensive metrics CSV for Hit@K
    metrics_path = results_dir / "rag_comprehensive_metrics.csv"
    
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found. Please run rag_visualization.py first.")
        return
    
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    
    # Rename index using label mapping
    new_index = [LABEL_MAPPING.get(idx, idx) for idx in metrics_df.index]
    metrics_df.index = new_index
    
    print(f"\nGenerating plots...")
    
    # Generate all three plots
    plot_hit_at_k_curves(metrics_df, output_dir)
    plot_rank_distribution(data_dict, output_dir)
    plot_failure_analysis(data_dict, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
