"""
SBERT-based evaluation for explanation generation tasks.
Handles both string-to-string and list-to-list comparisons.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class ExplanationEvaluator:
    """
    SBERT-based evaluator for explanation generation tasks.
    Supports both string-to-string and list-to-list comparisons.
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize the evaluator with a SentenceTransformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        print(f"✅ Loaded SBERT model: {model_name}")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_tensor=False)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self._get_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def _find_best_matches(self, list1: List[str], list2: List[str]) -> Tuple[List[Tuple[int, int, float]], float]:
        """
        Find best matches between two lists of strings using SBERT similarity.
        
        Args:
            list1: First list of strings
            list2: Second list of strings
            
        Returns:
            Tuple of (matches, average_similarity)
            matches: List of (index1, index2, similarity) tuples
        """
        if not list1 or not list2:
            return [], 0.0
        
        # Get embeddings for both lists
        embeddings1 = self._get_embeddings(list1)
        embeddings2 = self._get_embeddings(list2)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
        # Find best matches using greedy approach
        matches = []
        used_indices2 = set()
        
        # Sort by similarity (highest first)
        similarities = []
        for i in range(len(list1)):
            for j in range(len(list2)):
                similarities.append((i, j, similarity_matrix[i][j]))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, sim in similarities:
            if j not in used_indices2:
                matches.append((i, j, sim))
                used_indices2.add(j)
                if len(matches) == min(len(list1), len(list2)):
                    break
        
        # Calculate average similarity
        avg_similarity = np.mean([sim for _, _, sim in matches]) if matches else 0.0
        
        return matches, avg_similarity
    
    def evaluate_string_to_string(self, golden_explanation: str, predicted_explanation: str) -> Dict[str, float]:
        """
        Evaluate string-to-string explanation using SBERT.
        
        Args:
            golden_explanation: Ground truth explanation (string)
            predicted_explanation: Model prediction (string)
            
        Returns:
            Dictionary with similarity score
        """
        similarity = self._calculate_similarity(golden_explanation, predicted_explanation)
        
        return {
            'sbert_similarity': similarity,
            'evaluation_type': 'string_to_string'
        }
    
    def evaluate_list_to_list(self, golden_list: List[str], predicted_list: List[str]) -> Dict[str, float]:
        """
        Evaluate list-to-list explanations using SBERT with best matching.
        
        Args:
            golden_list: Ground truth explanations (list of strings)
            predicted_list: Model predictions (list of strings)
            
        Returns:
            Dictionary with similarity scores and match details
        """
        # Handle case where one is string and one is list
        if isinstance(golden_list, str) and isinstance(predicted_list, list):
            # Convert string to list by splitting on sentences
            golden_list = [golden_list]
        elif isinstance(golden_list, list) and isinstance(predicted_list, str):
            # Convert string to list by splitting on sentences
            predicted_list = [predicted_list]
        elif isinstance(golden_list, str) and isinstance(predicted_list, str):
            # Both are strings, convert to lists
            golden_list = [golden_list]
            predicted_list = [predicted_list]
        
        # Find best matches
        matches, avg_similarity = self._find_best_matches(golden_list, predicted_list)
        
        # Calculate individual similarities for matched pairs
        individual_similarities = [sim for _, _, sim in matches]
        
        # Calculate coverage (how many items were matched)
        coverage = len(matches) / max(len(golden_list), len(predicted_list)) if max(len(golden_list), len(predicted_list)) > 0 else 0.0
        
        return {
            'sbert_similarity': avg_similarity,
            'coverage': coverage,
            'num_matches': len(matches),
            'individual_similarities': individual_similarities,
            'matches': matches,
            'evaluation_type': 'list_to_list'
        }
    
    def evaluate_explanation(self, golden_explanation: Union[str, List[str]], 
                           predicted_explanation: Union[str, List[str]]) -> Dict[str, float]:
        """
        Universal evaluation method that handles both string and list inputs.
        
        Args:
            golden_explanation: Ground truth explanation (string or list)
            predicted_explanation: Model prediction (string or list)
            
        Returns:
            Dictionary with similarity scores
        """
        # Determine evaluation type
        golden_is_list = isinstance(golden_explanation, list)
        predicted_is_list = isinstance(predicted_explanation, list)
        
        if golden_is_list or predicted_is_list:
            # Use list-to-list evaluation
            return self.evaluate_list_to_list(golden_explanation, predicted_explanation)
        else:
            # Use string-to-string evaluation
            return self.evaluate_string_to_string(golden_explanation, predicted_explanation)
    
    def evaluate_batch(self, data: pd.DataFrame, 
                      golden_col: str, 
                      predicted_col: str,
                      evaluation_type: str = 'auto') -> pd.DataFrame:
        """
        Evaluate a batch of explanations.
        
        Args:
            data: DataFrame with golden and predicted explanations
            golden_col: Column name for golden explanations
            predicted_col: Column name for predicted explanations
            evaluation_type: 'auto', 'string_to_string', or 'list_to_list'
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for idx, row in data.iterrows():
            golden = row[golden_col]
            predicted = row[predicted_col]
            
            if evaluation_type == 'auto':
                result = self.evaluate_explanation(golden, predicted)
            elif evaluation_type == 'string_to_string':
                # Convert lists to strings if needed
                if isinstance(golden, list):
                    golden = ' '.join(golden)
                if isinstance(predicted, list):
                    predicted = ' '.join(predicted)
                result = self.evaluate_string_to_string(golden, predicted)
            elif evaluation_type == 'list_to_list':
                result = self.evaluate_list_to_list(golden, predicted)
            
            result['row_id'] = idx
            results.append(result)
        
        return pd.DataFrame(results)

def create_evaluation_summary(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Create summary statistics from evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'mean_sbert_similarity': results_df['sbert_similarity'].mean(),
        'std_sbert_similarity': results_df['sbert_similarity'].std(),
        'min_sbert_similarity': results_df['sbert_similarity'].min(),
        'max_sbert_similarity': results_df['sbert_similarity'].max(),
        'median_sbert_similarity': results_df['sbert_similarity'].median(),
    }
    
    # Add coverage statistics if available
    if 'coverage' in results_df.columns:
        summary.update({
            'mean_coverage': results_df['coverage'].mean(),
            'std_coverage': results_df['coverage'].std(),
            'mean_num_matches': results_df['num_matches'].mean(),
        })
    
    return summary