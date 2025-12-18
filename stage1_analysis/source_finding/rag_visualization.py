"""
Comprehensive RAG Source Finder Visualization and Analysis
===========================================================

This script loads existing RAG result CSV files and generates comprehensive
metrics and visualizations including:
- Ranking metrics (MRR, MAP, NDCG, Hit@K, Recall@K)
- Multiple golden answer analysis
- Embedding mode comparison
- Error analysis and failure patterns
- Publication-ready visualizations

Usage:
    python rag_visualization.py [--results_dir results/] [--output_dir results/visualizations/]
"""

import os
import sys
import argparse
import ast
import warnings
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style to match PARALLEL_PARC notebook
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.dpi'] = 100

# Define consistent color scheme
COLOR_STEELBLUE = 'steelblue'
COLOR_CORAL = 'coral'


class RAGDataLoader:
    """Load and parse RAG result CSV files"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.data = {}
        
    def load_all_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all RAG result CSV files from the results directory
        
        Returns:
            Dictionary mapping embedding mode to DataFrame
        """
        print(f"Loading RAG results from {self.results_dir}")
        
        # Find all rag_results_*.csv files
        rag_files = list(self.results_dir.glob("rag_results_*.csv"))
        
        if not rag_files:
            raise FileNotFoundError(f"No RAG result files found in {self.results_dir}")
        
        for file_path in rag_files:
            # Extract embedding mode from filename
            # e.g., rag_results_name_background.csv -> name_background
            mode = file_path.stem.replace("rag_results_", "")
            
            print(f"  Loading {file_path.name}...")
            df = pd.read_csv(file_path)
            
            # Parse list-formatted columns
            df = self._parse_list_columns(df)
            
            self.data[mode] = df
        
        print(f"Loaded {len(self.data)} embedding modes")
        return self.data
    
    def _parse_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse columns that contain list-formatted strings
        
        Args:
            df: DataFrame with string-formatted lists
            
        Returns:
            DataFrame with parsed lists
        """
        list_columns = ['top_k_sources', 'top_k_scores', 'all_golden_sources', 
                       'found_golden_sources', 'golden_ranks']
        
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._safe_parse_list)
        
        # Initialize missing columns with defaults for backward compatibility
        if 'all_golden_sources' not in df.columns:
            # Use gold_source as a single-item list
            df['all_golden_sources'] = df['gold_source'].apply(lambda x: [x] if pd.notna(x) else [])
        
        if 'num_golden_found' not in df.columns:
            # Calculate from gold_rank
            df['num_golden_found'] = df['gold_rank'].apply(lambda x: 1 if x > 0 else 0)
        
        if 'found_golden_sources' not in df.columns:
            # Use gold_source if found
            df['found_golden_sources'] = df.apply(
                lambda row: [row['gold_source']] if row['gold_rank'] > 0 and pd.notna(row['gold_source']) else [], 
                axis=1
            )
        
        if 'golden_ranks' not in df.columns:
            # Use gold_rank as a single-item list
            df['golden_ranks'] = df['gold_rank'].apply(lambda x: [x] if x > 0 else [])
        
        return df
    
    @staticmethod
    def _safe_parse_list(value):
        """Safely parse a string representation of a list"""
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []


class RankingMetrics:
    """Calculate standard Information Retrieval ranking metrics"""
    
    @staticmethod
    def mean_reciprocal_rank(gold_ranks: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            gold_ranks: List of ranks (1-indexed, -1 for not found)
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        for rank in gold_ranks:
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def hit_at_k(gold_ranks: List[int], k: int) -> float:
        """
        Calculate Hit@K (success rate at rank K)
        
        Args:
            gold_ranks: List of ranks (1-indexed, -1 for not found)
            k: Rank threshold
            
        Returns:
            Hit@K score (proportion of queries with gold in top-k)
        """
        hits = sum(1 for rank in gold_ranks if 0 < rank <= k)
        return hits / len(gold_ranks) if gold_ranks else 0.0
    
    @staticmethod
    def recall_at_k(df: pd.DataFrame, k: int) -> float:
        """
        Calculate Recall@K (proportion of all golden answers found in top-k)
        
        Args:
            df: DataFrame with 'num_golden_found' and 'all_golden_sources' columns
            k: Rank threshold
            
        Returns:
            Recall@K score
        """
        if 'num_golden_found' not in df.columns or 'all_golden_sources' not in df.columns:
            return 0.0
        
        total_golden = df['all_golden_sources'].apply(len).sum()
        found_golden = df['num_golden_found'].sum()
        
        return found_golden / total_golden if total_golden > 0 else 0.0
    
    @staticmethod
    def mean_average_precision(df: pd.DataFrame) -> float:
        """
        Calculate Mean Average Precision
        
        Args:
            df: DataFrame with 'golden_ranks' and 'all_golden_sources' columns
            
        Returns:
            MAP score
        """
        if 'golden_ranks' not in df.columns or 'all_golden_sources' not in df.columns:
            return 0.0
        
        average_precisions = []
        
        for _, row in df.iterrows():
            golden_ranks = row['golden_ranks']
            total_golden = len(row['all_golden_sources'])
            
            if total_golden == 0:
                continue
            
            if not golden_ranks or len(golden_ranks) == 0:
                average_precisions.append(0.0)
                continue
            
            # Calculate precision at each relevant position
            precisions = []
            for i, rank in enumerate(sorted(golden_ranks)):
                # Number of relevant items found so far / rank
                precision_at_rank = (i + 1) / rank
                precisions.append(precision_at_rank)
            
            # Average precision for this query
            ap = np.mean(precisions) if precisions else 0.0
            average_precisions.append(ap)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    @staticmethod
    def normalized_dcg(df: pd.DataFrame, k: int = 20) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain
        
        Args:
            df: DataFrame with 'golden_ranks' column
            k: Rank threshold
            
        Returns:
            NDCG score
        """
        if 'golden_ranks' not in df.columns:
            return 0.0
        
        ndcg_scores = []
        
        for _, row in df.iterrows():
            golden_ranks = row['golden_ranks']
            
            if not golden_ranks or len(golden_ranks) == 0:
                ndcg_scores.append(0.0)
                continue
            
            # Calculate DCG
            dcg = 0.0
            for rank in golden_ranks:
                if rank <= k:
                    # Relevance = 1 for golden answers, 0 otherwise
                    # DCG = sum(rel_i / log2(i + 1))
                    dcg += 1.0 / np.log2(rank + 1)
            
            # Calculate Ideal DCG (all golden answers at top positions)
            num_golden = len(golden_ranks)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_golden, k)))
            
            # Normalize
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    @staticmethod
    def calculate_all_metrics(df: pd.DataFrame, k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict:
        """
        Calculate all ranking metrics for a DataFrame
        
        Args:
            df: DataFrame with RAG results
            k_values: List of K values for Hit@K and Recall@K
            
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        # MRR
        metrics['MRR'] = RankingMetrics.mean_reciprocal_rank(df['gold_rank'].tolist())
        
        # MAP
        metrics['MAP'] = RankingMetrics.mean_average_precision(df)
        
        # NDCG
        metrics['NDCG@20'] = RankingMetrics.normalized_dcg(df, k=20)
        
        # Hit@K for various K values
        for k in k_values:
            metrics[f'Hit@{k}'] = RankingMetrics.hit_at_k(df['gold_rank'].tolist(), k)
        
        # Recall@K for various K values
        for k in k_values:
            metrics[f'Recall@{k}'] = RankingMetrics.recall_at_k(df, k)
        
        return metrics


class MultipleGoldenAnalyzer:
    """Analyze performance with multiple golden answers"""
    
    @staticmethod
    def golden_source_distribution(df: pd.DataFrame) -> pd.Series:
        """
        Get distribution of number of golden sources per target
        
        Args:
            df: DataFrame with 'all_golden_sources' column
            
        Returns:
            Series with counts of targets having N golden sources
        """
        if 'all_golden_sources' not in df.columns:
            return pd.Series()
        
        num_golden = df['all_golden_sources'].apply(len)
        return num_golden.value_counts().sort_index()
    
    @staticmethod
    def golden_coverage_stats(df: pd.DataFrame) -> Dict:
        """
        Calculate statistics about golden answer coverage
        
        Args:
            df: DataFrame with golden answer columns
            
        Returns:
            Dictionary of coverage statistics
        """
        if 'num_golden_found' not in df.columns or 'all_golden_sources' not in df.columns:
            return {}
        
        total_golden = df['all_golden_sources'].apply(len).sum()
        total_found = df['num_golden_found'].sum()
        
        # Coverage by number of golden answers available
        coverage_by_count = {}
        for num_golden in df['all_golden_sources'].apply(len).unique():
            subset = df[df['all_golden_sources'].apply(len) == num_golden]
            avg_found = subset['num_golden_found'].mean()
            coverage_by_count[num_golden] = avg_found
        
        return {
            'total_golden_sources': total_golden,
            'total_found': total_found,
            'overall_coverage': total_found / total_golden if total_golden > 0 else 0.0,
            'avg_golden_per_target': df['all_golden_sources'].apply(len).mean(),
            'avg_found_per_target': df['num_golden_found'].mean(),
            'coverage_by_count': coverage_by_count
        }
    
    @staticmethod
    def average_all_golden_ranks(df: pd.DataFrame) -> float:
        """
        Calculate average rank across ALL golden sources (not just best)
        
        Args:
            df: DataFrame with 'golden_ranks' column
            
        Returns:
            Average rank of all golden sources found
        """
        if 'golden_ranks' not in df.columns:
            return 0.0
        
        all_ranks = []
        for ranks in df['golden_ranks']:
            if ranks and len(ranks) > 0:
                all_ranks.extend(ranks)
        
        return np.mean(all_ranks) if all_ranks else 0.0


class ErrorAnalyzer:
    """Analyze errors and failure patterns"""
    
    @staticmethod
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
    
    @staticmethod
    def analyze_failures(df: pd.DataFrame) -> Dict:
        """
        Analyze cases where RAG failed to find the golden answer
        
        Args:
            df: DataFrame with RAG results
            
        Returns:
            Dictionary with failure analysis
        """
        # Performance categories
        df['performance_category'] = df['gold_rank'].apply(ErrorAnalyzer.categorize_performance)
        category_counts = df['performance_category'].value_counts()
        
        # Complete failures (not found in top-k)
        failures = df[df['gold_rank'] == -1]
        
        # Poor performance (found but rank > 10)
        poor_performance = df[df['gold_rank'] > 10]
        
        return {
            'total_examples': len(df),
            'complete_failures': len(failures),
            'failure_rate': len(failures) / len(df) if len(df) > 0 else 0.0,
            'poor_performance_count': len(poor_performance),
            'poor_performance_rate': len(poor_performance) / len(df) if len(df) > 0 else 0.0,
            'category_distribution': category_counts.to_dict(),
            'failure_examples': failures[['target', 'gold_source', 'predicted_rank_1']].head(10).to_dict('records') if len(failures) > 0 else []
        }
    
    @staticmethod
    def score_distribution_analysis(df: pd.DataFrame) -> Dict:
        """
        Analyze similarity score distributions for hits vs misses
        
        Args:
            df: DataFrame with 'gold_rank' and 'top_k_scores' columns
            
        Returns:
            Dictionary with score statistics
        """
        if 'top_k_scores' not in df.columns:
            return {}
        
        # Get top-1 scores
        df['top_1_score'] = df['top_k_scores'].apply(lambda x: x[0] if x and len(x) > 0 else 0.0)
        
        # Get scores for gold answers (when found)
        df['gold_score'] = df.apply(
            lambda row: row['top_k_scores'][row['gold_rank'] - 1] 
            if row['gold_rank'] > 0 and row['gold_rank'] <= len(row['top_k_scores']) 
            else np.nan,
            axis=1
        )
        
        hits = df[df['gold_rank'] == 1]
        misses = df[df['gold_rank'] != 1]
        
        return {
            'hit_score_mean': hits['top_1_score'].mean() if len(hits) > 0 else 0.0,
            'hit_score_std': hits['top_1_score'].std() if len(hits) > 0 else 0.0,
            'miss_score_mean': misses['top_1_score'].mean() if len(misses) > 0 else 0.0,
            'miss_score_std': misses['top_1_score'].std() if len(misses) > 0 else 0.0,
            'gold_score_mean': df['gold_score'].mean(),
            'gold_score_std': df['gold_score'].std(),
        }
    
    @staticmethod
    def confusion_analysis(df: pd.DataFrame, top_n: int = 10) -> Dict:
        """
        Analyze what was predicted instead of the gold answer
        
        Args:
            df: DataFrame with predictions
            top_n: Number of top confusion pairs to return
            
        Returns:
            Dictionary with confusion patterns
        """
        # Cases where prediction was wrong
        wrong_predictions = df[df['gold_rank'] != 1].copy()
        
        if len(wrong_predictions) == 0:
            return {'confusion_pairs': []}
        
        # Count (gold, predicted) pairs
        confusion_pairs = wrong_predictions.groupby(['gold_source', 'predicted_rank_1']).size()
        confusion_pairs = confusion_pairs.sort_values(ascending=False).head(top_n)
        
        return {
            'confusion_pairs': [
                {'gold': gold, 'predicted': pred, 'count': count}
                for (gold, pred), count in confusion_pairs.items()
            ]
        }


class RAGVisualizer:
    """Generate comprehensive visualizations"""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_performance_heatmap(self, metrics_df: pd.DataFrame, filename: str = "01_performance_heatmap.png"):
        """
        Create heatmap of metrics across embedding modes
        
        Args:
            metrics_df: DataFrame with metrics (rows=modes, columns=metrics)
            filename: Output filename
        """
        # Select key metrics for heatmap
        key_metrics = ['MRR', 'MAP', 'NDCG@20', 'Hit@1', 'Hit@5', 'Hit@10', 'Recall@10']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        if not available_metrics:
            print("No metrics available for heatmap")
            return
        
        heatmap_data = metrics_df[available_metrics].T  # Transpose: rows=metrics, cols=modes
        
        # Match notebook style exactly: dynamic figure size based on data shape
        plt.figure(figsize=(max(8, heatmap_data.shape[1]*0.6), max(6, heatmap_data.shape[0]*0.5)))
        
        # Use imshow exactly like domain pair heatmap (no extent, no vmin/vmax for count data)
        # But keep vmin/vmax for score data to ensure 0-1 range
        im = plt.imshow(heatmap_data.values, aspect="auto", cmap='viridis', vmin=0, vmax=1)
        
        plt.title('Performance Heatmap: Metrics × Embedding Modes', fontsize=14, fontweight='bold')
        plt.xlabel('Embedding Mode', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        
        # Disable grid lines to avoid white lines through the heatmap
        plt.grid(False)
        
        # Set ticks exactly like domain pair heatmap
        plt.xticks(range(heatmap_data.shape[1]), heatmap_data.columns.astype(str), rotation=45, ha="right")
        plt.yticks(range(heatmap_data.shape[0]), heatmap_data.index.astype(str))
        
        # Add text annotations with white color inside each cell (matching domain pair heatmap style)
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                value = heatmap_data.values[i, j]
                plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color='white', fontsize=9, fontweight='bold')
        
        # Add colorbar with exact notebook-style parameters
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Score")
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def plot_hit_at_k_curves(self, metrics_df: pd.DataFrame, filename: str = "02_hit_at_k_curves.png"):
        """
        Plot Hit@K curves for different K values
        
        Args:
            metrics_df: DataFrame with Hit@K metrics
            filename: Output filename
        """
        # Extract Hit@K columns
        hit_columns = [col for col in metrics_df.columns if col.startswith('Hit@')]
        
        if not hit_columns:
            print("No Hit@K metrics available")
            return
        
        # Extract K values
        k_values = sorted([int(col.split('@')[1]) for col in hit_columns])
        
        plt.figure(figsize=(10, 6))
        
        # Use alternating colors for modes
        colors = [COLOR_STEELBLUE, COLOR_CORAL] * ((len(metrics_df.index) // 2) + 1)
        for idx, mode in enumerate(metrics_df.index):
            hit_scores = [metrics_df.loc[mode, f'Hit@{k}'] for k in k_values]
            plt.plot(k_values, hit_scores, marker='o', label=mode, linewidth=2, color=colors[idx])
        
        plt.xlabel('K (Rank Threshold)', fontsize=12)
        plt.ylabel('Hit@K (Success Rate)', fontsize=12)
        plt.title('Hit@K: Success Rate at Different Rank Thresholds', fontsize=14, fontweight='bold')
        plt.legend(title='Embedding Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def plot_rank_distribution(self, data_dict: Dict[str, pd.DataFrame], filename: str = "03_rank_distribution.png"):
        """
        Plot histogram of gold ranks by mode
        
        Args:
            data_dict: Dictionary mapping mode to DataFrame
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        modes = list(data_dict.keys())[:4]  # Plot up to 4 modes
        
        for idx, mode in enumerate(modes):
            df = data_dict[mode]
            
            # Filter out -1 (not found) for histogram
            found_ranks = df[df['gold_rank'] > 0]['gold_rank']
            
            if len(found_ranks) > 0:
                axes[idx].hist(found_ranks, bins=20, edgecolor='black', alpha=0.7, color=COLOR_STEELBLUE)
                axes[idx].axvline(found_ranks.median(), color=COLOR_CORAL, linestyle='--', 
                                 linewidth=2, label=f'Median: {found_ranks.median():.1f}')
                axes[idx].set_xlabel('Gold Rank', fontsize=12)
                axes[idx].set_ylabel('Frequency', fontsize=12)
                axes[idx].set_title(f'{mode}\n(Found: {len(found_ranks)}/{len(df)})', fontsize=14, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            else:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                axes[idx].set_title(f'{mode}', fontsize=14, fontweight='bold')
        
        plt.suptitle('Distribution of Gold Ranks by Embedding Mode', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def plot_metric_comparison(self, metrics_df: pd.DataFrame, filename: str = "04_mrr_map_ndcg_comparison.png"):
        """
        Bar chart comparison of MRR, MAP, NDCG
        
        Args:
            metrics_df: DataFrame with metrics
            filename: Output filename
        """
        metrics_to_plot = ['MRR', 'MAP', 'NDCG@20']
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        if not available_metrics:
            print("No MRR/MAP/NDCG metrics available")
            return
        
        plot_data = metrics_df[available_metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(plot_data.index))
        width = 0.25
        
        # Use steelblue and coral for metrics
        bar_colors = [COLOR_STEELBLUE, COLOR_CORAL, 'lightsteelblue']  # fallback for 3rd metric
        for i, metric in enumerate(available_metrics):
            offset = width * (i - len(available_metrics) / 2 + 0.5)
            color = bar_colors[i] if i < len(bar_colors) else COLOR_STEELBLUE
            ax.bar(x + offset, plot_data[metric], width, label=metric, alpha=0.8, color=color)
        
        ax.set_xlabel('Embedding Mode', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('MRR, MAP, and NDCG@20 Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def plot_golden_coverage(self, data_dict: Dict[str, pd.DataFrame], filename: str = "05_golden_answer_coverage.png"):
        """
        Visualize how many golden answers are found vs available
        
        Args:
            data_dict: Dictionary mapping mode to DataFrame
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Average coverage rate
        coverage_data = []
        for mode, df in data_dict.items():
            if 'num_golden_found' in df.columns and 'all_golden_sources' in df.columns:
                stats = MultipleGoldenAnalyzer.golden_coverage_stats(df)
                coverage_data.append({
                    'mode': mode,
                    'coverage': stats.get('overall_coverage', 0.0),
                    'avg_found': stats.get('avg_found_per_target', 0.0),
                    'avg_available': stats.get('avg_golden_per_target', 0.0)
                })
        
        if coverage_data:
            coverage_df = pd.DataFrame(coverage_data)
            
            x = np.arange(len(coverage_df))
            width = 0.35
            
            bars1 = axes[0].bar(x - width/2, coverage_df['avg_available'], width, label='Available', alpha=0.8, color=COLOR_CORAL)
            bars2 = axes[0].bar(x + width/2, coverage_df['avg_found'], width, label='Found', alpha=0.8, color=COLOR_STEELBLUE)
            
            axes[0].set_xlabel('Embedding Mode', fontsize=12)
            axes[0].set_ylabel('Average per Target', fontsize=12)
            axes[0].set_title('Gold Sources: Available vs Found', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(coverage_df['mode'], rotation=45, ha='right')
            # Move legend outside the plot area to avoid overlap
            axes[0].legend(loc='upper left', bbox_to_anchor=(0, 1))
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Subplot 2: Coverage rate
            axes[1].bar(coverage_df['mode'], coverage_df['coverage'], alpha=0.8, color=COLOR_STEELBLUE)
            axes[1].set_xlabel('Embedding Mode', fontsize=12)
            axes[1].set_ylabel('Coverage Rate', fontsize=12)
            axes[1].set_title('Overall Gold Answer Coverage', fontsize=14, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].set_ylim([0, 1.0])
        
        plt.suptitle('Multiple Gold Answer Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def plot_score_distribution(self, data_dict: Dict[str, pd.DataFrame], filename: str = "06_score_distribution.png"):
        """
        Plot similarity score distributions for hits vs misses
        
        Args:
            data_dict: Dictionary mapping mode to DataFrame
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        modes = list(data_dict.keys())[:4]
        
        for idx, mode in enumerate(modes):
            df = data_dict[mode].copy()
            
            if 'top_k_scores' not in df.columns:
                continue
            
            # Get top-1 scores
            df['top_1_score'] = df['top_k_scores'].apply(lambda x: x[0] if x and len(x) > 0 else 0.0)
            
            # Separate hits and misses
            hits = df[df['gold_rank'] == 1]['top_1_score']
            misses = df[df['gold_rank'] != 1]['top_1_score']
            
            if len(hits) > 0:
                axes[idx].hist(hits, bins=20, alpha=0.7, label=f'Hits (n={len(hits)})', 
                              color=COLOR_STEELBLUE, edgecolor='black')
            if len(misses) > 0:
                axes[idx].hist(misses, bins=20, alpha=0.7, label=f'Misses (n={len(misses)})', 
                              color=COLOR_CORAL, edgecolor='black')
            
            axes[idx].set_xlabel('Top-1 Similarity Score', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{mode}', fontsize=14, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Similarity Score Distribution: Rank-1 Hits vs Misses', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def plot_failure_analysis(self, data_dict: Dict[str, pd.DataFrame], filename: str = "07_failure_analysis.png"):
        """
        Visualize failure categories and frequencies
        
        Args:
            data_dict: Dictionary mapping mode to DataFrame
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        modes = list(data_dict.keys())[:4]
        
        for idx, mode in enumerate(modes):
            df = data_dict[mode].copy()
            
            # Categorize performance
            df['category'] = df['gold_rank'].apply(ErrorAnalyzer.categorize_performance)
            
            category_counts = df['category'].value_counts()
            
            # Sort categories in logical order
            category_order = ['Perfect (Rank 1)', 'Excellent (Rank 2-3)', 'Good (Rank 4-5)', 
                            'Fair (Rank 6-10)', 'Poor (Rank >10)', 'Not Found']
            category_counts = category_counts.reindex([c for c in category_order if c in category_counts.index], 
                                                     fill_value=0)
            
            # Use gradient from steelblue to coral for performance categories
            num_cats = len(category_counts)
            if num_cats > 0:
                colors_list = [COLOR_STEELBLUE] * (num_cats // 2) + [COLOR_CORAL] * ((num_cats + 1) // 2)
                # Adjust for better gradient: perfect=steelblue, not found=coral
                if num_cats >= 6:
                    colors_list = [COLOR_STEELBLUE, 'lightsteelblue', 'lightblue', 'lightsalmon', COLOR_CORAL, 'darkred']
                elif num_cats >= 4:
                    colors_list = [COLOR_STEELBLUE, 'lightsteelblue', 'lightsalmon', COLOR_CORAL]
                colors_list = colors_list[:num_cats]
            else:
                colors_list = [COLOR_STEELBLUE]
            
            axes[idx].bar(range(len(category_counts)), category_counts.values, 
                         color=colors_list, alpha=0.7, edgecolor='black')
            axes[idx].set_xticks(range(len(category_counts)))
            axes[idx].set_xticklabels(category_counts.index, rotation=45, ha='right', fontsize=10)
            axes[idx].set_ylabel('Count', fontsize=12)
            axes[idx].set_title(f'{mode}', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Performance Categories by Embedding Mode', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    def plot_recall_at_k_curves(self, metrics_df: pd.DataFrame, filename: str = "08_recall_at_k_curves.png"):
        """
        Plot Recall@K curves for different K values
        
        Args:
            metrics_df: DataFrame with Recall@K metrics
            filename: Output filename
        """
        # Extract Recall@K columns
        recall_columns = [col for col in metrics_df.columns if col.startswith('Recall@')]
        
        if not recall_columns:
            print("No Recall@K metrics available")
            return
        
        # Extract K values
        k_values = sorted([int(col.split('@')[1]) for col in recall_columns])
        
        plt.figure(figsize=(10, 6))
        
        # Use alternating colors for modes
        colors = [COLOR_STEELBLUE, COLOR_CORAL] * ((len(metrics_df.index) // 2) + 1)
        for idx, mode in enumerate(metrics_df.index):
            recall_scores = [metrics_df.loc[mode, f'Recall@{k}'] for k in k_values]
            plt.plot(k_values, recall_scores, marker='s', label=mode, linewidth=2, color=colors[idx])
        
        plt.xlabel('K (Rank Threshold)', fontsize=12)
        plt.ylabel('Recall@K', fontsize=12)
        plt.title('Recall@K: Proportion of Gold Answers Found at Different Thresholds', 
                 fontsize=14, fontweight='bold')
        plt.legend(title='Embedding Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


class ComprehensiveAnalyzer:
    """Main analyzer that orchestrates all analysis and visualization"""
    
    def __init__(self, results_dir: str = "results", output_dir: str = "results/visualizations"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.loader = RAGDataLoader(results_dir)
        self.visualizer = RAGVisualizer(output_dir)
        self.data = {}
        self.metrics = {}
        
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("RAG SOURCE FINDER - COMPREHENSIVE ANALYSIS")
        print("="*80 + "\n")
        
        # 1. Load data
        print("Step 1: Loading data...")
        self.data = self.loader.load_all_results()
        print(f"✓ Loaded {len(self.data)} embedding modes\n")
        
        # 2. Calculate metrics
        print("Step 2: Calculating ranking metrics...")
        self.calculate_all_metrics()
        print("✓ Metrics calculated\n")
        
        # 3. Generate visualizations
        print("Step 3: Generating visualizations...")
        self.generate_all_visualizations()
        print("✓ Visualizations complete\n")
        
        # 4. Save comprehensive metrics
        print("Step 4: Saving comprehensive metrics...")
        self.save_comprehensive_metrics()
        print("✓ Metrics saved\n")
        
        # 5. Generate summary report
        print("Step 5: Generating summary report...")
        self.generate_summary_report()
        print("✓ Summary report generated\n")
        
        print("="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  Metrics: {self.results_dir}/rag_comprehensive_metrics.csv")
        print(f"  Visualizations: {self.output_dir}/")
        print(f"  Summary: {self.results_dir}/rag_analysis_summary.txt")
    
    def calculate_all_metrics(self):
        """Calculate all metrics for all embedding modes"""
        for mode, df in self.data.items():
            print(f"  Calculating metrics for: {mode}")
            self.metrics[mode] = RankingMetrics.calculate_all_metrics(df)
    
    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        metrics_df = pd.DataFrame(self.metrics).T
        
        print("  Generating visualizations...")
        
        # 1. Performance heatmap
        self.visualizer.plot_performance_heatmap(metrics_df)
        
        # 2. Hit@K curves
        self.visualizer.plot_hit_at_k_curves(metrics_df)
        
        # 3. Rank distribution
        self.visualizer.plot_rank_distribution(self.data)
        
        # 4. MRR/MAP/NDCG comparison
        self.visualizer.plot_metric_comparison(metrics_df)
        
        # 5. Golden answer coverage
        self.visualizer.plot_golden_coverage(self.data)
        
        # 6. Score distribution
        self.visualizer.plot_score_distribution(self.data)
        
        # 7. Failure analysis
        self.visualizer.plot_failure_analysis(self.data)
        
        # 8. Recall@K curves
        self.visualizer.plot_recall_at_k_curves(metrics_df)
    
    def save_comprehensive_metrics(self):
        """Save all metrics to CSV"""
        metrics_df = pd.DataFrame(self.metrics).T
        
        # Add additional statistics
        for mode, df in self.data.items():
            # Golden answer stats
            golden_stats = MultipleGoldenAnalyzer.golden_coverage_stats(df)
            metrics_df.loc[mode, 'Total_Golden_Sources'] = golden_stats.get('total_golden_sources', 0)
            metrics_df.loc[mode, 'Total_Found'] = golden_stats.get('total_found', 0)
            metrics_df.loc[mode, 'Golden_Coverage'] = golden_stats.get('overall_coverage', 0.0)
            metrics_df.loc[mode, 'Avg_Golden_Per_Target'] = golden_stats.get('avg_golden_per_target', 0.0)
            metrics_df.loc[mode, 'Avg_Found_Per_Target'] = golden_stats.get('avg_found_per_target', 0.0)
            metrics_df.loc[mode, 'Avg_All_Golden_Ranks'] = MultipleGoldenAnalyzer.average_all_golden_ranks(df)
            
            # Error analysis
            error_stats = ErrorAnalyzer.analyze_failures(df)
            metrics_df.loc[mode, 'Failure_Rate'] = error_stats.get('failure_rate', 0.0)
            metrics_df.loc[mode, 'Poor_Performance_Rate'] = error_stats.get('poor_performance_rate', 0.0)
            
            # Score analysis
            score_stats = ErrorAnalyzer.score_distribution_analysis(df)
            metrics_df.loc[mode, 'Hit_Score_Mean'] = score_stats.get('hit_score_mean', 0.0)
            metrics_df.loc[mode, 'Miss_Score_Mean'] = score_stats.get('miss_score_mean', 0.0)
            metrics_df.loc[mode, 'Gold_Score_Mean'] = score_stats.get('gold_score_mean', 0.0)
        
        output_path = Path(self.results_dir) / "rag_comprehensive_metrics.csv"
        metrics_df.to_csv(output_path)
        print(f"  Saved comprehensive metrics to: {output_path}")
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        output_path = Path(self.results_dir) / "rag_analysis_summary.txt"
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RAG SOURCE FINDER - ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of embedding modes analyzed: {len(self.data)}\n")
            f.write(f"Embedding modes: {', '.join(self.data.keys())}\n")
            
            for mode, df in self.data.items():
                f.write(f"\n  {mode}: {len(df)} examples\n")
            
            # Best performing mode by metric
            f.write("\n\nBEST PERFORMING MODES BY METRIC\n")
            f.write("-"*80 + "\n")
            
            metrics_df = pd.DataFrame(self.metrics).T
            
            key_metrics = ['MRR', 'MAP', 'NDCG@20', 'Hit@1', 'Hit@5', 'Hit@10']
            for metric in key_metrics:
                if metric in metrics_df.columns:
                    best_mode = metrics_df[metric].idxmax()
                    best_value = metrics_df[metric].max()
                    f.write(f"  {metric:15s}: {best_mode:30s} ({best_value:.4f})\n")
            
            # Detailed metrics per mode
            f.write("\n\nDETAILED METRICS BY EMBEDDING MODE\n")
            f.write("="*80 + "\n")
            
            for mode, df in self.data.items():
                f.write(f"\n{mode}\n")
                f.write("-"*80 + "\n")
                
                # Ranking metrics
                mode_metrics = self.metrics[mode]
                f.write(f"  MRR:        {mode_metrics.get('MRR', 0):.4f}\n")
                f.write(f"  MAP:        {mode_metrics.get('MAP', 0):.4f}\n")
                f.write(f"  NDCG@20:    {mode_metrics.get('NDCG@20', 0):.4f}\n")
                f.write(f"  Hit@1:      {mode_metrics.get('Hit@1', 0):.4f}\n")
                f.write(f"  Hit@5:      {mode_metrics.get('Hit@5', 0):.4f}\n")
                f.write(f"  Hit@10:     {mode_metrics.get('Hit@10', 0):.4f}\n")
                f.write(f"  Recall@10:  {mode_metrics.get('Recall@10', 0):.4f}\n")
                
                # Golden answer coverage
                golden_stats = MultipleGoldenAnalyzer.golden_coverage_stats(df)
                f.write(f"\n  Gold Answer Coverage:\n")
                f.write(f"    Total golden sources: {golden_stats.get('total_golden_sources', 0)}\n")
                f.write(f"    Total found:          {golden_stats.get('total_found', 0)}\n")
                f.write(f"    Coverage rate:        {golden_stats.get('overall_coverage', 0):.4f}\n")
                f.write(f"    Avg per target:       {golden_stats.get('avg_golden_per_target', 0):.2f}\n")
                f.write(f"    Avg found:            {golden_stats.get('avg_found_per_target', 0):.2f}\n")
                
                # Error analysis
                error_stats = ErrorAnalyzer.analyze_failures(df)
                f.write(f"\n  Error Analysis:\n")
                f.write(f"    Failure rate:         {error_stats.get('failure_rate', 0):.4f}\n")
                f.write(f"    Poor performance:     {error_stats.get('poor_performance_rate', 0):.4f}\n")
                
                # Performance distribution
                f.write(f"\n  Performance Distribution:\n")
                for category, count in error_stats.get('category_distribution', {}).items():
                    f.write(f"    {category:25s}: {count:4d} ({count/len(df)*100:5.1f}%)\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            
            # Best overall mode
            best_overall_mode = metrics_df['MRR'].idxmax()
            f.write(f"\nBest Overall Mode (by MRR): {best_overall_mode}\n")
            f.write(f"  This mode achieved the highest Mean Reciprocal Rank ({metrics_df.loc[best_overall_mode, 'MRR']:.4f})\n")
            
            # Mode with best coverage
            coverage_data = {mode: MultipleGoldenAnalyzer.golden_coverage_stats(df).get('overall_coverage', 0) 
                           for mode, df in self.data.items()}
            best_coverage_mode = max(coverage_data, key=coverage_data.get)
            f.write(f"\nBest Coverage Mode: {best_coverage_mode}\n")
            f.write(f"  This mode found the highest proportion of all gold answers ({coverage_data[best_coverage_mode]:.4f})\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"  Saved summary report to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive RAG Source Finder Analysis and Visualization"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing RAG result CSV files (default: results)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations",
        help="Directory for output visualizations (default: results/visualizations)"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ComprehensiveAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    try:
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

