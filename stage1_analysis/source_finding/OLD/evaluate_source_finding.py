"""
Evaluation Framework for Source Finding
Computes metrics: Exact Match, Top-K Accuracy, MRR, Average Rank
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    method: str
    model: Optional[str]
    exact_match_acc: float
    top_1_acc: float
    top_3_acc: float
    top_5_acc: float
    top_10_acc: float
    mrr: float
    avg_rank: float
    num_samples: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'method': self.method,
            'model': self.model,
            'exact_match_acc': self.exact_match_acc,
            'top_1_acc': self.top_1_acc,
            'top_3_acc': self.top_3_acc,
            'top_5_acc': self.top_5_acc,
            'top_10_acc': self.top_10_acc,
            'mrr': self.mrr,
            'avg_rank': self.avg_rank,
            'num_samples': self.num_samples
        }


class SourceFindingEvaluator:
    """
    Evaluate source finding methods with multiple metrics
    """
    
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    def evaluate_rag_results(
        self, 
        results_df: pd.DataFrame,
        method_name: str = "RAG"
    ) -> EvaluationMetrics:
        """
        Evaluate RAG results
        
        Args:
            results_df: DataFrame from RAGSourceFinder.evaluate_on_dataset
            method_name: Name for this method
            
        Returns:
            EvaluationMetrics object
        """
        n = len(results_df)
        
        # Exact match (rank 1 matches gold)
        exact_match = (results_df['predicted_rank_1'] == results_df['gold_source']).sum()
        exact_match_acc = exact_match / n
        
        # Top-K accuracy
        top_1 = (results_df['gold_rank'] == 1).sum() / n
        top_3 = (results_df['gold_rank'].between(1, 3)).sum() / n
        top_5 = (results_df['gold_rank'].between(1, 5)).sum() / n
        top_10 = (results_df['gold_rank'].between(1, 10)).sum() / n
        
        # MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for rank in results_df['gold_rank']:
            if rank > 0:  # -1 means not found
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        mrr = np.mean(reciprocal_ranks)
        
        # Average rank (only for found items)
        valid_ranks = results_df[results_df['gold_rank'] > 0]['gold_rank']
        avg_rank = valid_ranks.mean() if len(valid_ranks) > 0 else -1
        
        return EvaluationMetrics(
            method=method_name,
            model=None,  # RAG doesn't use a model for ranking
            exact_match_acc=exact_match_acc,
            top_1_acc=top_1,
            top_3_acc=top_3,
            top_5_acc=top_5,
            top_10_acc=top_10,
            mrr=mrr,
            avg_rank=avg_rank,
            num_samples=n
        )
    
    def evaluate_iterative_results(
        self,
        results_df: pd.DataFrame,
        method_name: str,
        model_name: str,
        top_k_list: List[int] = [1, 3, 5, 10]
    ) -> EvaluationMetrics:
        """
        Evaluate iterative (tournament/sequential) results
        
        Args:
            results_df: DataFrame with columns: id, target, gold_source, predicted_source, ranked_sources
            method_name: 'Tournament' or 'Sequential'
            model_name: Name of LLM used
            top_k_list: List of K values to evaluate
            
        Returns:
            EvaluationMetrics object
        """
        n = len(results_df)
        
        # Exact match
        exact_match = (results_df['predicted_source'] == results_df['gold_source']).sum()
        exact_match_acc = exact_match / n
        
        # For iterative methods, we need to compute ranking if available
        # If 'ranked_sources' column exists (list of sources in preference order)
        if 'ranked_sources' in results_df.columns:
            ranks = []
            for _, row in results_df.iterrows():
                ranked = row['ranked_sources']
                gold = row['gold_source']
                if gold in ranked:
                    ranks.append(ranked.index(gold) + 1)
                else:
                    ranks.append(-1)
            
            results_df['gold_rank'] = ranks
            
            # Top-K accuracies
            top_1 = (results_df['gold_rank'] == 1).sum() / n
            top_3 = (results_df['gold_rank'].between(1, 3)).sum() / n
            top_5 = (results_df['gold_rank'].between(1, 5)).sum() / n
            top_10 = (results_df['gold_rank'].between(1, 10)).sum() / n
            
            # MRR
            reciprocal_ranks = [1.0/r if r > 0 else 0.0 for r in results_df['gold_rank']]
            mrr = np.mean(reciprocal_ranks)
            
            # Average rank
            valid_ranks = results_df[results_df['gold_rank'] > 0]['gold_rank']
            avg_rank = valid_ranks.mean() if len(valid_ranks) > 0 else -1
        else:
            # If no ranking available, only compute exact match
            top_1 = exact_match_acc
            top_3 = exact_match_acc
            top_5 = exact_match_acc
            top_10 = exact_match_acc
            mrr = exact_match_acc
            avg_rank = 1.0 if exact_match_acc == 1.0 else -1
        
        return EvaluationMetrics(
            method=method_name,
            model=model_name,
            exact_match_acc=exact_match_acc,
            top_1_acc=top_1,
            top_3_acc=top_3,
            top_5_acc=top_5,
            top_10_acc=top_10,
            mrr=mrr,
            avg_rank=avg_rank,
            num_samples=n
        )
    
    def compare_methods(
        self, 
        metrics_list: List[EvaluationMetrics]
    ) -> pd.DataFrame:
        """
        Create comparison table of multiple methods
        
        Args:
            metrics_list: List of EvaluationMetrics objects
            
        Returns:
            DataFrame with comparison
        """
        rows = []
        for m in metrics_list:
            rows.append(m.to_dict())
        
        df = pd.DataFrame(rows)
        
        # Reorder columns
        col_order = ['method', 'model', 'exact_match_acc', 'top_1_acc', 'top_3_acc', 
                     'top_5_acc', 'top_10_acc', 'mrr', 'avg_rank', 'num_samples']
        df = df[col_order]
        
        return df
    
    def visualize_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Create visualization comparing methods
        
        Args:
            comparison_df: DataFrame from compare_methods
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Create method labels (method + model if available)
        comparison_df['label'] = comparison_df.apply(
            lambda row: f"{row['method']}\n({row['model']})" if pd.notna(row['model']) else row['method'],
            axis=1
        )
        
        # 1. Exact Match Accuracy
        ax = axes[0]
        bars = ax.bar(range(len(comparison_df)), comparison_df['exact_match_acc'])
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['label'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Exact Match Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # 2. Top-K Accuracy
        ax = axes[1]
        x = np.arange(len(comparison_df))
        width = 0.2
        
        ax.bar(x - 1.5*width, comparison_df['top_1_acc'], width, label='Top-1', alpha=0.8)
        ax.bar(x - 0.5*width, comparison_df['top_3_acc'], width, label='Top-3', alpha=0.8)
        ax.bar(x + 0.5*width, comparison_df['top_5_acc'], width, label='Top-5', alpha=0.8)
        ax.bar(x + 1.5*width, comparison_df['top_10_acc'], width, label='Top-10', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['label'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Top-K Accuracy')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. MRR and Average Rank
        ax = axes[2]
        ax2 = ax.twinx()
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, comparison_df['mrr'], width, label='MRR', alpha=0.8, color='steelblue')
        bars2 = ax2.bar(x + width/2, comparison_df['avg_rank'], width, label='Avg Rank', alpha=0.8, color='coral')
        
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['label'], rotation=45, ha='right')
        ax.set_ylabel('MRR', color='steelblue')
        ax.set_title('MRR and Average Rank')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='y', labelcolor='steelblue')
        
        ax2.set_ylabel('Average Rank', color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def analyze_error_patterns(
        self,
        results_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Analyze common error patterns
        
        Args:
            results_df: Results DataFrame with predictions
            corpus_df: Full SCAR corpus for domain info
            top_n: Number of top errors to show
            
        Returns:
            DataFrame with error analysis
        """
        # Identify mismatches
        errors = results_df[results_df['predicted_rank_1'] != results_df['gold_source']].copy()
        
        if len(errors) == 0:
            print("No errors found!")
            return pd.DataFrame()
        
        # Get domain info
        domain_map = corpus_df.set_index('system_b')['system_b_domain'].to_dict()
        
        errors['gold_domain'] = errors['gold_source'].map(domain_map)
        errors['predicted_domain'] = errors['predicted_rank_1'].map(domain_map)
        
        # Count error patterns
        error_patterns = errors.groupby(['gold_source', 'predicted_rank_1']).size().reset_index(name='count')
        error_patterns = error_patterns.sort_values('count', ascending=False).head(top_n)
        
        return error_patterns
    
    def export_results(
        self,
        comparison_df: pd.DataFrame,
        save_dir: str,
        prefix: str = "evaluation"
    ):
        """
        Export evaluation results
        
        Args:
            comparison_df: Comparison DataFrame
            save_dir: Directory to save results
            prefix: Prefix for filenames
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(save_dir, f"{prefix}_metrics.csv")
        comparison_df.to_csv(csv_path, index=False)
        print(f"Saved metrics to {csv_path}")
        
        # Save JSON
        json_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(comparison_df.to_dict(orient='records'), f, indent=2)
        print(f"Saved metrics to {json_path}")
        
        # Save summary stats
        summary_path = os.path.join(save_dir, f"{prefix}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Source Finding Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for _, row in comparison_df.iterrows():
                f.write(f"Method: {row['method']}")
                if pd.notna(row['model']):
                    f.write(f" (Model: {row['model']})")
                f.write("\n")
                f.write(f"  Exact Match Acc: {row['exact_match_acc']:.4f}\n")
                f.write(f"  Top-1 Acc:       {row['top_1_acc']:.4f}\n")
                f.write(f"  Top-3 Acc:       {row['top_3_acc']:.4f}\n")
                f.write(f"  Top-5 Acc:       {row['top_5_acc']:.4f}\n")
                f.write(f"  Top-10 Acc:      {row['top_10_acc']:.4f}\n")
                f.write(f"  MRR:             {row['mrr']:.4f}\n")
                f.write(f"  Avg Rank:        {row['avg_rank']:.2f}\n")
                f.write(f"  Samples:         {row['num_samples']}\n")
                f.write("\n")
        
        print(f"Saved summary to {summary_path}")


def print_metrics_table(comparison_df: pd.DataFrame):
    """
    Pretty print metrics table
    
    Args:
        comparison_df: Comparison DataFrame
    """
    print("\n" + "="*80)
    print("SOURCE FINDING EVALUATION RESULTS")
    print("="*80 + "\n")
    
    # Format for display
    display_df = comparison_df.copy()
    
    # Format percentages
    for col in ['exact_match_acc', 'top_1_acc', 'top_3_acc', 'top_5_acc', 'top_10_acc', 'mrr']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
    
    display_df['avg_rank'] = display_df['avg_rank'].apply(lambda x: f"{x:.2f}")
    
    # Print
    print(display_df.to_string(index=False))
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("Source Finding Evaluation Framework")
    print("=" * 50)
    print("\nUse this module to evaluate and compare source finding methods")

