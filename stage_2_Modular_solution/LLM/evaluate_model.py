"""
Evaluate Model Script - Evaluate generated analogies using LLM-as-judge
Includes semantic similarity matching for gold source identification
"""

import argparse
import sys
import os
import json
import time
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm

# Load environment variables from .env file BEFORE other imports
from dotenv import load_dotenv
env_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..', '.env'),  # ../../.env (root)
    os.path.join(os.path.dirname(__file__), '.env'),  # ./env (local)
]
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage1_analysis', 'mapping_generation'))

import dspy
from easy_llm_importer import LLMClient, DSPyAdapter

from config import (
    JUDGE_MODEL,
    JUDGE_INSTRUCTIONS,
    TEST_MODE_RECORD_LIMIT,
    RESULTS_DIR,
)
from precompute_similarity import SemanticMatcher, find_top1_by_embedding


# =============================================================================
# DSPy SIGNATURE FOR LLM-AS-JUDGE
# =============================================================================

class AnalogyJudgeSignature(dspy.Signature):
    __doc__ = JUDGE_INSTRUCTIONS
    
    target_concept: str = dspy.InputField(desc="The unfamiliar target concept being explained")
    chosen_analogy: str = dspy.InputField(desc="The source concept chosen as the analogy")
    selection_reasoning: str = dspy.InputField(desc="The reasoning used to select this analogy (empty if not available)")
    
    analogy_coherence: int = dspy.OutputField(desc="Score 1-3: Does the pairing make intuitive sense?")
    mapping_soundness: int = dspy.OutputField(desc="Score 1-3: Could source properties map to target properties?")
    explanatory_power: int = dspy.OutputField(desc="Score 1-3: Would this help a learner understand the target?")


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_analogy(
    judge: dspy.Module,
    target: str, 
    chosen_analogy: str, 
    reasoning: str = "",
    max_retries: int = 3
) -> Dict:
    """
    Evaluate a single analogy using the LLM judge.
    Returns exact format from LLM_as_judge.ipynb.
    
    Args:
        judge: DSPy judge module
        target: Target concept
        chosen_analogy: The selected analogy
        reasoning: Selection reasoning (if available)
        max_retries: Number of retry attempts
        
    Returns:
        dict with exact format:
            - analogy_coherence: int (1-3)
            - mapping_soundness: int (1-3)
            - explanatory_power: int (1-3)
            - average_score: float (rounded to 2 decimals)
            - judge_reasoning: str
            - status: "success" | "error"
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = judge(
                target_concept=target,
                chosen_analogy=chosen_analogy,
                selection_reasoning=reasoning if reasoning else "No reasoning provided"
            )
            
            # Extract scores (ensure they are integers in 1-3 range)
            coherence = int(result.analogy_coherence)
            mapping = int(result.mapping_soundness)
            explanatory = int(result.explanatory_power)
            
            # Clamp to valid range (1-3)
            coherence = max(1, min(3, coherence))
            mapping = max(1, min(3, mapping))
            explanatory = max(1, min(3, explanatory))
            
            # Calculate average
            avg_score = (coherence + mapping + explanatory) / 3
            
            return {
                "analogy_coherence": coherence,
                "mapping_soundness": mapping,
                "explanatory_power": explanatory,
                "average_score": round(avg_score, 2),
                "judge_reasoning": result.reasoning,
                "status": "success"
            }
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)  # Exponential backoff
    
    # All retries failed
    return {
        "analogy_coherence": None,
        "mapping_soundness": None,
        "explanatory_power": None,
        "average_score": None,
        "judge_reasoning": f"ERROR: {last_error}",
        "status": "error"
    }


def compute_gold_matching(
    generated_analogies: List[str],
    all_gold_sources: List[str],
    semantic_matcher: SemanticMatcher
) -> Dict:
    """
    Compute exact and semantic gold source matching.
    
    Compares ONLY source concept names (no sub-concepts).
    Unified evaluation for both targetonly and withsub modes.
    
    Args:
        generated_analogies: List of generated analogy concepts (source names only)
        all_gold_sources: List of ALL gold sources for this target
        semantic_matcher: SemanticMatcher instance
        
    Returns:
        Dict with gold_rank, sem_gold_source, sem_gold_rank, similarity_scores, etc.
    """
    return semantic_matcher.find_semantic_match(
        generated_analogies, 
        all_gold_sources
    )


def evaluate_model_results(
    input_path: str,
    test_mode: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Evaluate all results from a model generation run.
    
    Args:
        input_path: Path to the generation results CSV
        test_mode: If True, only process limited records
        verbose: Whether to print detailed output
        
    Returns:
        DataFrame with evaluation results
    """
    print("=" * 70)
    print(f"Evaluating: {input_path}")
    print(f"Test Mode: {test_mode}")
    print("=" * 70)
    
    # Load generation results
    print(f"\nLoading generation results...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    
    # Limit records in test mode
    if test_mode:
        df = df.head(TEST_MODE_RECORD_LIMIT)
        print(f"Test mode: Using first {len(df)} records")
        verbose = True
    
    # Initialize LLM judge
    print(f"\nInitializing LLM judge with {JUDGE_MODEL}...")
    client = LLMClient()
    adapter = DSPyAdapter(client, JUDGE_MODEL)
    lm = adapter.get_dspy_lm()
    dspy.configure(lm=lm)
    
    judge = dspy.ChainOfThought(AnalogyJudgeSignature)
    
    # Initialize semantic matcher
    print("Loading semantic matcher...")
    semantic_matcher = SemanticMatcher()
    
    # Process all records
    results = []
    print(f"\nEvaluating {len(df)} records...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Parse generated analogies
        try:
            generated_analogies = json.loads(row['generated_analogies'])
        except (json.JSONDecodeError, KeyError):
            generated_analogies = []
        
        # Parse generated sub-concepts if available (from withsub mode)
        analogy_subconcepts = None
        if 'analogy_subconcepts' in row and pd.notna(row['analogy_subconcepts']):
            try:
                analogy_subconcepts = json.loads(row['analogy_subconcepts'])
            except (json.JSONDecodeError, KeyError):
                analogy_subconcepts = None
        
        # Get target sub-concepts if available
        target_subconcepts = None
        if 'sub_concepts' in row and pd.notna(row['sub_concepts']):
            target_subconcepts = row['sub_concepts']
        
        # Get all gold sources (required)
        all_gold_sources = []
        if 'all_gold_sources' in row and pd.notna(row['all_gold_sources']):
            try:
                all_gold_sources = json.loads(row['all_gold_sources'])
            except (json.JSONDecodeError, KeyError):
                all_gold_sources = []
        
        target = row['target']
        
        # Top-1 Baseline: First generated analogy (model's order)
        top1_baseline = generated_analogies[0] if generated_analogies else ""
        
        # Determine if we're in withsub mode (have sub-concepts for analogies)
        is_withsub_mode = analogy_subconcepts is not None and len(analogy_subconcepts) > 0
        
        # Top-1 Embedding: Best by OpenAI embedding similarity (target vs analogies)
        # In withsub mode, includes sub-concepts in the comparison
        embedding_result = find_top1_by_embedding(
            target=target,
            target_subconcepts=target_subconcepts,
            generated_analogies=generated_analogies,
            analogy_subconcepts=analogy_subconcepts,
            use_subconcepts=is_withsub_mode
        )
        top1_embedding = embedding_result['top1_embedding']
        top1_embedding_score = embedding_result['top1_embedding_score']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Record ID: {row['id']}")
            print(f"Target: {target}")
            if target_subconcepts:
                print(f"Target Sub-concepts: {target_subconcepts}")
            print(f"Gold Sources ({len(all_gold_sources)}): {all_gold_sources}")
            print(f"Top-1 Baseline (model order): {top1_baseline}")
            print(f"Top-1 Embedding (by similarity): {top1_embedding} (score: {top1_embedding_score})")
            print(f"All Generated: {generated_analogies}")
            if analogy_subconcepts:
                print(f"Generated Sub-concepts: {analogy_subconcepts}")
        
        # Evaluate with LLM judge - TOP1_BASELINE
        eval_baseline = evaluate_analogy(
            judge=judge,
            target=target,
            chosen_analogy=top1_baseline,
            reasoning=""
        )
        
        # Evaluate with LLM judge - TOP1_EMBEDDING
        # Only evaluate if top1_embedding is different from top1_baseline
        if top1_embedding and top1_embedding != top1_baseline:
            eval_embedding = evaluate_analogy(
                judge=judge,
                target=target,
                chosen_analogy=top1_embedding,
                reasoning=""
            )
        else:
            # Same analogy, reuse the baseline result
            eval_embedding = eval_baseline.copy()
        
        # Compute gold matching (source concepts only - unified for both modes)
        gold_match = compute_gold_matching(
            generated_analogies=generated_analogies,
            all_gold_sources=all_gold_sources,
            semantic_matcher=semantic_matcher
        )
        
        if verbose:
            print(f"\nLLM Judge Scores (for top1_baseline: '{top1_baseline}'):")
            print(f"  Coherence: {eval_baseline['analogy_coherence']}")
            print(f"  Mapping: {eval_baseline['mapping_soundness']}")
            print(f"  Explanatory: {eval_baseline['explanatory_power']}")
            print(f"  Average: {eval_baseline['average_score']}")
            print(f"\nLLM Judge Scores (for top1_embedding: '{top1_embedding}'):")
            print(f"  Coherence: {eval_embedding['analogy_coherence']}")
            print(f"  Mapping: {eval_embedding['mapping_soundness']}")
            print(f"  Explanatory: {eval_embedding['explanatory_power']}")
            print(f"  Average: {eval_embedding['average_score']}")
            print(f"\nGold Matching:")
            print(f"  Exact Ranks (all): {gold_match['gold_ranks_list']}")
            print(f"  Semantic Ranks (all): {gold_match['sem_gold_ranks_list']}")
            print(f"  Found Generated Analogies (exact match): {gold_match['found_gold_sources']}")
            print(f"  Found Generated Analogies (semantic match): {gold_match['sem_gold_sources']}")
            print(f"  Exact Ranks by generated analogy: {gold_match['gold_ranks']}")
            print(f"  Semantic Ranks by generated analogy: {gold_match['sem_gold_ranks']}")
            print(f"\nSimilarity per Gold Source:")
            for gs, stats in gold_match['similarity_per_gold'].items():
                print(f"  {gs}: highest={stats['highest']}, avg={stats['avg']}")
        
        # Build judge result dictionaries
        judge_baseline = {
            'analogy': top1_baseline,
            'coherence': eval_baseline['analogy_coherence'],
            'mapping': eval_baseline['mapping_soundness'],
            'explanatory': eval_baseline['explanatory_power'],
            'average': eval_baseline['average_score'],
            'reasoning': eval_baseline['judge_reasoning'],
            'status': eval_baseline['status']
        }
        
        judge_embedding = {
            'analogy': top1_embedding,
            'coherence': eval_embedding['analogy_coherence'],
            'mapping': eval_embedding['mapping_soundness'],
            'explanatory': eval_embedding['explanatory_power'],
            'average': eval_embedding['average_score'],
            'reasoning': eval_embedding['judge_reasoning'],
            'status': eval_embedding['status']
        }
        
        # Combine all results
        result_record = {
            'id': row['id'],
            'target': target,
            'target_subconcepts': target_subconcepts if target_subconcepts else '',
            'all_gold_sources': json.dumps(all_gold_sources),
            'num_gold_sources': len(all_gold_sources),
            'generated_analogies': row['generated_analogies'],
            'generated_subconcepts': json.dumps(analogy_subconcepts) if analogy_subconcepts else '',  # From withsub mode
            'reasoning': row['reasoning'],
            # Top-1 selections
            'top1_baseline': top1_baseline,  # First in model's output order
            'top1_embedding': top1_embedding,  # Best by OpenAI embedding similarity to target
            'top1_embedding_score': top1_embedding_score,  # Similarity score for top1_embedding
            'embedding_all_scores': json.dumps(embedding_result['all_scores']),  # All analogy -> target similarity scores
            # LLM Judge scores - stored as dictionaries
            'judge_baseline': json.dumps(judge_baseline),  # Dict with all judge scores for top1_baseline
            'judge_embedding': json.dumps(judge_embedding),  # Dict with all judge scores for top1_embedding
            'status': eval_baseline['status'],
            # Exact match results
            'gold_ranks_list': json.dumps(gold_match['gold_ranks_list']),  # All exact ranks in order [1, 3, 5]
            'gold_ranks': json.dumps(gold_match['gold_ranks']),  # Dict: generated_analogy -> rank
            # Semantic match results  
            'sem_gold_ranks_list': json.dumps(gold_match['sem_gold_ranks_list']),  # All semantic ranks in order [2, 4]
            'sem_gold_ranks': json.dumps(gold_match['sem_gold_ranks']),  # Dict: generated_analogy -> semantic rank
            # Per-gold similarity stats
            'similarity_per_gold': json.dumps(gold_match['similarity_per_gold']),  # Dict: gold -> {scores, highest, avg}
        }
        results.append(result_record)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute and display summary statistics
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    successful = results_df[results_df['status'] == 'success']
    print(f"Total records: {len(results_df)}")
    print(f"Successful evaluations: {len(successful)}")
    print(f"Errors: {len(results_df) - len(successful)}")
    
    if len(successful) > 0:
        # Helper to extract scores from judge JSON
        def get_judge_score(judge_json: str, key: str) -> float:
            try:
                judge = json.loads(judge_json)
                return judge.get(key, 0)
            except:
                return 0
        
        successful = successful.copy()
        
        # Extract baseline judge scores
        successful['baseline_coherence'] = successful['judge_baseline'].apply(lambda x: get_judge_score(x, 'coherence'))
        successful['baseline_mapping'] = successful['judge_baseline'].apply(lambda x: get_judge_score(x, 'mapping'))
        successful['baseline_explanatory'] = successful['judge_baseline'].apply(lambda x: get_judge_score(x, 'explanatory'))
        successful['baseline_avg'] = successful['judge_baseline'].apply(lambda x: get_judge_score(x, 'average'))
        
        # Extract embedding judge scores
        successful['embedding_coherence'] = successful['judge_embedding'].apply(lambda x: get_judge_score(x, 'coherence'))
        successful['embedding_mapping'] = successful['judge_embedding'].apply(lambda x: get_judge_score(x, 'mapping'))
        successful['embedding_explanatory'] = successful['judge_embedding'].apply(lambda x: get_judge_score(x, 'explanatory'))
        successful['embedding_avg'] = successful['judge_embedding'].apply(lambda x: get_judge_score(x, 'average'))
        
        print(f"\nLLM Judge Scores - TOP1_BASELINE (mean):")
        print(f"  Coherence:   {successful['baseline_coherence'].mean():.2f}")
        print(f"  Mapping:     {successful['baseline_mapping'].mean():.2f}")
        print(f"  Explanatory: {successful['baseline_explanatory'].mean():.2f}")
        print(f"  Average:     {successful['baseline_avg'].mean():.2f}")
        
        print(f"\nLLM Judge Scores - TOP1_EMBEDDING (mean):")
        print(f"  Coherence:   {successful['embedding_coherence'].mean():.2f}")
        print(f"  Mapping:     {successful['embedding_mapping'].mean():.2f}")
        print(f"  Explanatory: {successful['embedding_explanatory'].mean():.2f}")
        print(f"  Average:     {successful['embedding_avg'].mean():.2f}")
        
        # Helper to get best rank from ranks list
        def get_best_rank(ranks_json: str) -> int:
            """Get best (lowest) rank from JSON list, or -1 if empty."""
            try:
                ranks = json.loads(ranks_json)
                return min(ranks) if ranks else -1
            except:
                return -1
        
        # Hit@K exact - uses best rank from all gold sources
        results_df['best_exact_rank'] = results_df['gold_ranks_list'].apply(get_best_rank)
        total = len(results_df)
        hit1_exact = (results_df['best_exact_rank'] == 1).sum() / total
        hit2_exact = (results_df['best_exact_rank'].between(1, 2)).sum() / total
        hit3_exact = (results_df['best_exact_rank'].between(1, 3)).sum() / total
        
        print(f"\nHit@K (Exact Match - any gold source):")
        print(f"  Hit@1: {hit1_exact:.1%}")
        print(f"  Hit@2: {hit2_exact:.1%}")
        print(f"  Hit@3: {hit3_exact:.1%}")
        
        # Hit@K semantic - uses best rank from all gold sources
        results_df['best_sem_rank'] = results_df['sem_gold_ranks_list'].apply(get_best_rank)
        
        # Effective rank: exact if found, else semantic
        def get_effective_rank(row):
            if row['best_exact_rank'] != -1:
                return row['best_exact_rank']
            return row['best_sem_rank']
        
        effective_rank = results_df.apply(get_effective_rank, axis=1)
        hit1_sem = (effective_rank == 1).sum() / total
        hit2_sem = (effective_rank.between(1, 2)).sum() / total
        hit3_sem = (effective_rank.between(1, 3)).sum() / total
        
        print(f"\nHit@K (Semantic, threshold >= 0.5):")
        print(f"  Hit@1: {hit1_sem:.1%}")
        print(f"  Hit@2: {hit2_sem:.1%}")
        print(f"  Hit@3: {hit3_sem:.1%}")
        
        # Top-1 Embedding similarity stats
        avg_embedding_score = results_df['top1_embedding_score'].mean()
        print(f"\nTop-1 Embedding (Target-Analogy Similarity):")
        print(f"  Average Score: {avg_embedding_score:.4f}")
    
    # Save results
    output_path = input_path.replace('.csv', '_eval.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-generated analogies")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the generation results CSV file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited records and verbose output"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each record"
    )
    
    args = parser.parse_args()
    
    evaluate_model_results(
        input_path=args.input,
        test_mode=args.test,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
