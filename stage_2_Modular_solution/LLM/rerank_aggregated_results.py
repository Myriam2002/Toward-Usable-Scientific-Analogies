"""
Rerank Aggregated Results Script

Reranks generated analogies using an LLM reranker and re-evaluates the top-1 choice
with LLM-as-a-judge. Supports both targetonly (with subconcept backfill) and withsub modes.

Usage:
    Terminal A: python rerank_aggregated_results.py --input results/all_results_targetonly.csv
    Terminal B: python rerank_aggregated_results.py --input results/all_results_withsub.csv
    
    Optional flags:
    --test N    : Process only first N rows
    --resume    : Skip rows already present in existing rerank output
"""

import argparse
import sys
import os
import json
import time
import ast
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
    MAPPING_MODEL,
    SCAR_PATH,
    TEST_MODE_RECORD_LIMIT,
    RESULTS_DIR,
    NUM_ANALOGIES,
)
from evaluate_model import AnalogyJudgeSignature, evaluate_analogy
from run_model import (
    SourcePropertyMappingSignature,
    extract_subconcepts,
    parse_mapped_properties,
)


# =============================================================================
# RERANKER CONFIGURATION
# =============================================================================

RERANKER_MODEL = "meta-llama-3-1-70b-instruct"

RERANKER_INSTRUCTIONS = """You are an expert at ranking scientific analogies.

Given a target concept and a list of candidate analogies, rank them from BEST to WORST.

A good analogy:
- Uses a FAMILIAR source concept to explain an UNFAMILIAR target concept
- Has meaningful structural or functional parallels between source and target
- Helps learners build understanding through comparison
- Does not require prior knowledge to understand

Your task:
1. Analyze the target concept and its properties (if provided)
2. Review each candidate analogy
3. Rank ALL candidates from best (most effective analogy) to worst (least effective)

Return a JSON list of exactly {num_analogies} analogy strings, ordered from BEST to WORST.
The list must contain ALL {num_analogies} candidates, just reordered by quality.

Example output format (JSON):
["best_analogy", "second_best", "third_best", ..., "worst_analogy"]
""".format(num_analogies=NUM_ANALOGIES)


# =============================================================================
# DSPy SIGNATURE FOR RERANKER
# =============================================================================

class AnalogyRerankerSignature(dspy.Signature):
    """Rank candidate analogies from best to worst for a target concept."""
    
    __doc__ = RERANKER_INSTRUCTIONS
    
    target_concept: str = dspy.InputField(desc="The unfamiliar target concept to find analogies for")
    target_properties: str = dspy.InputField(desc="Comma-separated properties/sub-concepts of the target (empty if not available)")
    candidate_analogies: str = dspy.InputField(desc="List of candidate analogies to rank (one per line)")
    ranked_analogies: str = dspy.OutputField(desc=f"JSON list of exactly {NUM_ANALOGIES} analogies, ranked from best to worst")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_scar_dataset() -> pd.DataFrame:
    """Load SCAR dataset for subconcept extraction."""
    print(f"Loading SCAR dataset from {SCAR_PATH}...")
    df = pd.read_csv(SCAR_PATH)
    print(f"Loaded {len(df)} records")
    return df


def get_scar_record_for_target(df_scar: pd.DataFrame, target: str) -> Optional[pd.Series]:
    """Get first SCAR record matching the target concept."""
    matches = df_scar[df_scar['system_a'] == target]
    if len(matches) > 0:
        return matches.iloc[0]
    return None


def generate_subconcepts_for_targetonly(
    target: str,
    generated_analogies: List[str],
    df_scar: pd.DataFrame,
    verbose: bool = False
) -> tuple[str, List[str]]:
    """
    Generate target_subconcepts and generated_subconcepts for targetonly mode.
    
    Returns:
        tuple: (target_subconcepts_str, generated_subconcepts_list)
    """
    # Get SCAR record for this target
    scar_record = get_scar_record_for_target(df_scar, target)
    if scar_record is None:
        if verbose:
            print(f"  Warning: No SCAR record found for target '{target}', using empty subconcepts")
        return "", [""] * len(generated_analogies)
    
    # Extract target subconcepts
    target_subconcepts = extract_subconcepts(scar_record)
    if not target_subconcepts:
        if verbose:
            print(f"  Warning: No subconcepts found for target '{target}'")
        return "", [""] * len(generated_analogies)
    
    # Get target description
    target_description = scar_record.get('system_a_background', '')
    
    # Parse target properties into list
    target_props = [p.strip() for p in target_subconcepts.split(",") if p.strip()]
    
    if verbose:
        print(f"  Target subconcepts: {target_subconcepts}")
        print(f"  Generating mappings for {len(generated_analogies)} analogies...")
    
    # Initialize mapping model
    mapping_client = LLMClient()
    mapping_adapter = DSPyAdapter(mapping_client, MAPPING_MODEL)
    mapping_lm = mapping_adapter.get_dspy_lm()
    
    # Generate subconcepts for each analogy
    generated_subconcepts = []
    
    with dspy.context(lm=mapping_lm):
        mapper = dspy.Predict(SourcePropertyMappingSignature)
        
        for analogy in generated_analogies:
            try:
                mapping_result = mapper(
                    unfamiliar_concept=target,
                    description_of_unfamiliar_concept=target_description if target_description else "",
                    properties_of_unfamiliar_concept=target_props,
                    familiar_concept=analogy
                )
                
                # Convert dict to comma-separated string
                mapped_props = mapping_result.mapped_source_properties
                mapped_string = parse_mapped_properties(mapped_props, target_props)
                generated_subconcepts.append(mapped_string)
                
            except Exception as e:
                if verbose:
                    print(f"    Warning: Mapping failed for '{analogy}': {e}")
                generated_subconcepts.append("")
    
    return target_subconcepts, generated_subconcepts


def rerank_analogies(
    reranker: dspy.Module,
    target: str,
    target_subconcepts: str,
    generated_analogies: List[str],
    max_retries: int = 3
) -> tuple[Optional[List[str]], str]:
    """
    Rerank analogies using LLM reranker.
    
    Returns:
        tuple: (ranked_list, reasoning_str) on success, (None, error_msg) on failure
    """
    # Format candidate list
    candidates_text = "\n".join([f"{i+1}. {analogy}" for i, analogy in enumerate(generated_analogies)])
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = reranker(
                target_concept=target,
                target_properties=target_subconcepts if target_subconcepts else "",
                candidate_analogies=candidates_text
            )
            
            # Extract reasoning if available
            reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
            
            # Parse ranked_analogies (should be JSON string)
            ranked_str = result.ranked_analogies.strip()
            
            # Try to extract JSON from the response (might have extra text)
            # Look for JSON array pattern
            import re
            json_match = re.search(r'\[.*\]', ranked_str, re.DOTALL)
            if json_match:
                ranked_str = json_match.group(0)
            
            # Parse JSON
            try:
                ranked_list = json.loads(ranked_str)
            except json.JSONDecodeError:
                # Try Python literal eval as fallback
                ranked_list = ast.literal_eval(ranked_str)
            
            # Validate: must be a list
            if not isinstance(ranked_list, list):
                raise ValueError(f"Expected list, got {type(ranked_list)}")
            
            # Validate: must contain all analogies (fuzzy match)
            ranked_list_lower = [str(a).lower().strip() for a in ranked_list]
            original_lower = [str(a).lower().strip() for a in generated_analogies]
            
            # Check if all original analogies are present (allowing for slight variations)
            missing = []
            validated_ranked = []
            
            for ranked_item in ranked_list:
                ranked_str_lower = str(ranked_item).lower().strip()
                # Try exact match first
                if ranked_str_lower in original_lower:
                    idx = original_lower.index(ranked_str_lower)
                    validated_ranked.append(generated_analogies[idx])
                else:
                    # Try fuzzy match (contains or is contained)
                    found = False
                    for orig_idx, orig_lower in enumerate(original_lower):
                        if ranked_str_lower in orig_lower or orig_lower in ranked_str_lower:
                            validated_ranked.append(generated_analogies[orig_idx])
                            found = True
                            break
                    if not found:
                        missing.append(ranked_item)
            
            # If we have missing items, try to fill them from remaining originals
            remaining_originals = [a for a in generated_analogies if a not in validated_ranked]
            if missing and remaining_originals:
                # Fill missing with remaining originals (preserve order as much as possible)
                for orig in generated_analogies:
                    if orig not in validated_ranked:
                        validated_ranked.append(orig)
            
            # Ensure we have exactly NUM_ANALOGIES items
            if len(validated_ranked) < NUM_ANALOGIES:
                # Add remaining originals
                for orig in generated_analogies:
                    if len(validated_ranked) >= NUM_ANALOGIES:
                        break
                    if orig not in validated_ranked:
                        validated_ranked.append(orig)
            elif len(validated_ranked) > NUM_ANALOGIES:
                validated_ranked = validated_ranked[:NUM_ANALOGIES]
            
            return validated_ranked, reasoning
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)  # Exponential backoff
    
    return None, f"ERROR: {last_error or 'Failed after all retries'}"


def compute_rerank_indices(
    ranked_analogies: List[str],
    original_analogies: List[str]
) -> List[int]:
    """
    Compute indices mapping reranked items back to original positions (1-indexed).
    
    Returns:
        List of integers (1-20) indicating original positions
    """
    indices = []
    original_lower = [str(a).lower().strip() for a in original_analogies]
    
    for ranked_item in ranked_analogies:
        ranked_lower = str(ranked_item).lower().strip()
        try:
            idx = original_lower.index(ranked_lower)
            indices.append(idx + 1)  # 1-indexed
        except ValueError:
            # Fuzzy match fallback
            for orig_idx, orig_lower in enumerate(original_lower):
                if ranked_lower in orig_lower or orig_lower in ranked_lower:
                    indices.append(orig_idx + 1)
                    break
            else:
                # If still not found, use -1 as marker
                indices.append(-1)
    
    return indices


# =============================================================================
# MAIN RERANKING FUNCTION
# =============================================================================

def rerank_aggregated_results(
    input_path: str,
    test_mode: bool = False,
    test_limit: Optional[int] = None,
    resume: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Rerank analogies in aggregated results CSV and re-evaluate with LLM-as-judge.
    
    Args:
        input_path: Path to aggregated results CSV
        test_mode: If True, only process limited records
        test_limit: Number of records to process in test mode (defaults to TEST_MODE_RECORD_LIMIT)
        resume: If True, skip rows already present in existing rerank output
        verbose: Whether to print detailed output
        
    Returns:
        DataFrame with reranking results
    """
    print("=" * 70)
    print(f"Reranking: {input_path}")
    print(f"Test Mode: {test_mode}")
    print(f"Resume: {resume}")
    print("=" * 70)
    
    # Load input CSV
    print(f"\nLoading input CSV...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    
    # Determine mode from filename or data
    is_targetonly = "targetonly" in input_path.lower()
    if not is_targetonly:
        # Check if mode column exists
        if 'mode' in df.columns:
            is_targetonly = (df['mode'].iloc[0] == 'targetonly') if len(df) > 0 else False
    
    print(f"Mode: {'targetonly' if is_targetonly else 'withsub'}")
    
    # Check for existing output (for resume or incremental saves)
    output_path = input_path.replace('.csv', '_rerank.csv')
    existing_ids = set()
    results = []  # Start with empty results list
    
    # Load existing results if file exists (for resume or to continue from previous run)
    if os.path.exists(output_path):
        print(f"\nLoading existing results from {output_path}...")
        try:
            existing_df = pd.read_csv(output_path)
            existing_ids = set(existing_df['id'].astype(str))
            # Convert DataFrame to list of dicts for incremental appending
            results = existing_df.to_dict('records')
            print(f"Found {len(existing_ids)} existing records - will skip these and continue")
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}")
            results = []
    elif resume:
        print(f"\nResume mode requested but no existing file found at {output_path}")
        print("Starting fresh run...")
    
    # Limit records in test mode
    if test_mode:
        limit = test_limit if test_limit is not None else TEST_MODE_RECORD_LIMIT
        df = df.head(limit)
        print(f"Test mode: Processing first {len(df)} records")
        verbose = True
    
    # Load SCAR dataset (needed for target_subconcepts extraction in both modes)
    df_scar = load_scar_dataset()
    
    # Initialize reranker LLM
    print(f"\nInitializing reranker with {RERANKER_MODEL}...")
    reranker_client = LLMClient()
    reranker_adapter = DSPyAdapter(reranker_client, RERANKER_MODEL)
    reranker_lm = reranker_adapter.get_dspy_lm()
    
    reranker = dspy.ChainOfThought(AnalogyRerankerSignature)
    
    # Initialize judge LLM
    print(f"Initializing judge with {JUDGE_MODEL}...")
    judge_client = LLMClient()
    judge_adapter = DSPyAdapter(judge_client, JUDGE_MODEL)
    judge_lm = judge_adapter.get_dspy_lm()
    
    judge = dspy.ChainOfThought(AnalogyJudgeSignature)
    
    # Process all records
    print(f"\nProcessing {len(df)} records...")
    print(f"Starting with {len(results)} already processed records")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Reranking"):
        record_id = str(row['id'])
        
        # Skip if already processed (from existing file)
        if record_id in existing_ids:
            if verbose:
                print(f"Skipping record {record_id} (already processed)")
            continue
        
        target = row['target']
        
        # Parse generated analogies
        try:
            generated_analogies = json.loads(row['generated_analogies'])
        except (json.JSONDecodeError, KeyError, TypeError):
            if verbose:
                print(f"  Warning: Could not parse generated_analogies for record {record_id}")
            generated_analogies = []
        
        if not generated_analogies:
            if verbose:
                print(f"  Skipping record {record_id}: No analogies to rerank")
            continue
        
        # Get or generate subconcepts
        sec_generated_subconcepts = None  # Will store generated subconcepts for targetonly
        
        # Always extract target_subconcepts from SCAR dataset (for both modes)
        scar_record = get_scar_record_for_target(df_scar, target)
        if scar_record is not None:
            target_subconcepts = extract_subconcepts(scar_record)
        else:
            # Fallback: try to get from CSV if SCAR lookup fails
            target_subconcepts = row.get('target_subconcepts', '')
            if pd.isna(target_subconcepts):
                target_subconcepts = ''
            if verbose and not target_subconcepts:
                print(f"  Warning: No SCAR record found for target '{target}', using empty subconcepts")
        
        if is_targetonly:
            # Generate subconcepts for each analogy
            if target_subconcepts:
                # Generate per-analogy subconcepts using mapping model
                target_description = scar_record.get('system_a_background', '') if scar_record is not None else ''
                target_props = [p.strip() for p in target_subconcepts.split(",") if p.strip()]
                
                if verbose:
                    print(f"  Target subconcepts: {target_subconcepts}")
                    print(f"  Generating mappings for {len(generated_analogies)} analogies...")
                
                # Initialize mapping model
                mapping_client = LLMClient()
                mapping_adapter = DSPyAdapter(mapping_client, MAPPING_MODEL)
                mapping_lm = mapping_adapter.get_dspy_lm()
                
                # Generate subconcepts for each analogy
                generated_subconcepts = []
                
                with dspy.context(lm=mapping_lm):
                    mapper = dspy.Predict(SourcePropertyMappingSignature)
                    
                    for analogy in generated_analogies:
                        try:
                            mapping_result = mapper(
                                unfamiliar_concept=target,
                                description_of_unfamiliar_concept=target_description if target_description else "",
                                properties_of_unfamiliar_concept=target_props,
                                familiar_concept=analogy
                            )
                            
                            # Convert dict to comma-separated string
                            mapped_props = mapping_result.mapped_source_properties
                            mapped_string = parse_mapped_properties(mapped_props, target_props)
                            generated_subconcepts.append(mapped_string)
                            
                        except Exception as e:
                            if verbose:
                                print(f"    Warning: Mapping failed for '{analogy}': {e}")
                            generated_subconcepts.append("")
                
                # Store for saving in sec_generated_subconcepts column
                sec_generated_subconcepts = generated_subconcepts
            else:
                generated_subconcepts = [""] * len(generated_analogies)
                sec_generated_subconcepts = generated_subconcepts
        else:
            # Use existing generated_subconcepts from CSV
            try:
                generated_subconcepts_str = row.get('generated_subconcepts', '')
                if pd.notna(generated_subconcepts_str) and generated_subconcepts_str:
                    generated_subconcepts = json.loads(generated_subconcepts_str)
                else:
                    generated_subconcepts = [""] * len(generated_analogies)
            except (json.JSONDecodeError, TypeError):
                generated_subconcepts = [""] * len(generated_analogies)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Record ID: {record_id}")
            print(f"Target: {target}")
            if target_subconcepts:
                print(f"Target Subconcepts: {target_subconcepts}")
            print(f"Generated Analogies ({len(generated_analogies)}): {generated_analogies[:3]}...")
        
        # Rerank analogies (use reranker LM context)
        with dspy.context(lm=reranker_lm):
            ranked_analogies, rerank_reasoning = rerank_analogies(
                reranker=reranker,
                target=target,
                target_subconcepts=target_subconcepts,
                generated_analogies=generated_analogies,
                max_retries=3
            )
        
        if ranked_analogies is None:
            if verbose:
                print(f"  Error: Reranking failed: {rerank_reasoning}")
            # Create error result
            result_record = row.to_dict()
            result_record.update({
                'target_subconcepts': target_subconcepts,  # Always include extracted subconcepts
                'llm_rerank_order': json.dumps([]),
                'llm_rerank_order_indices': json.dumps([]),
                'top1_rerank': '',
                'rerank_reasoning': rerank_reasoning,
                'judge_rerank': json.dumps({
                    'analogy': '',
                    'coherence': None,
                    'mapping': None,
                    'explanatory': None,
                    'average': None,
                    'reasoning': f"Reranking failed: {rerank_reasoning}",
                    'status': 'error'
                })
            })
            # Add sec_generated_subconcepts for targetonly (even if reranking failed)
            if is_targetonly and sec_generated_subconcepts is not None:
                result_record['sec_generated_subconcepts'] = json.dumps(sec_generated_subconcepts)
            results.append(result_record)
            continue
        
        # Get top-1 reranked
        top1_rerank = ranked_analogies[0] if ranked_analogies else ""
        
        # Compute rerank indices
        rerank_indices = compute_rerank_indices(ranked_analogies, generated_analogies)
        
        if verbose:
            print(f"Top-1 Reranked: {top1_rerank}")
            print(f"Rerank Order (first 5): {ranked_analogies[:5]}")
        
        # Evaluate top-1 reranked with LLM-as-judge (use judge LM context)
        with dspy.context(lm=judge_lm):
            eval_result = evaluate_analogy(
                judge=judge,
                target=target,
                chosen_analogy=top1_rerank,
                reasoning=rerank_reasoning,
                max_retries=3
            )
        
        # Build judge_rerank dict (same format as judge_baseline/judge_embedding)
        judge_rerank = {
            'analogy': top1_rerank,
            'coherence': eval_result['analogy_coherence'],
            'mapping': eval_result['mapping_soundness'],
            'explanatory': eval_result['explanatory_power'],
            'average': eval_result['average_score'],
            'reasoning': eval_result['judge_reasoning'],
            'status': eval_result['status']
        }
        
        if verbose:
            print(f"Judge Scores: coherence={judge_rerank['coherence']}, "
                  f"mapping={judge_rerank['mapping']}, "
                  f"explanatory={judge_rerank['explanatory']}, "
                  f"average={judge_rerank['average']}")
        
        # Build result record (copy all original columns, add new ones)
        result_record = row.to_dict()
        result_record.update({
            'target_subconcepts': target_subconcepts,  # Always update with extracted subconcepts
            'llm_rerank_order': json.dumps(ranked_analogies),
            'llm_rerank_order_indices': json.dumps(rerank_indices),
            'top1_rerank': top1_rerank,
            'rerank_reasoning': rerank_reasoning,
            'judge_rerank': json.dumps(judge_rerank)
        })
        # Add sec_generated_subconcepts for targetonly mode
        if is_targetonly and sec_generated_subconcepts is not None:
            result_record['sec_generated_subconcepts'] = json.dumps(sec_generated_subconcepts)
        
        # Add to results and save immediately (incremental checkpoint)
        results.append(result_record)
        
        # Save after each record to prevent data loss
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            if verbose or (idx + 1) % 10 == 0:
                print(f"  ✓ Saved {len(results_df)} records (checkpoint)")
        except Exception as save_error:
            print(f"  ⚠ Warning: Failed to save checkpoint: {save_error}")
    
    # Final save (redundant but ensures consistency)
    results_df = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print(f"Final save to: {output_path}")
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} total records")
    print("=" * 70)
    
    # Summary statistics
    if len(results_df) > 0:
        successful = results_df[results_df['judge_rerank'].apply(
            lambda x: json.loads(x)['status'] == 'success' if isinstance(x, str) else False
        )]
        print(f"\nSummary:")
        print(f"  Total processed: {len(results_df)}")
        print(f"  Successful evaluations: {len(successful)}")
        print(f"  Errors: {len(results_df) - len(successful)}")
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rerank aggregated LLM results and re-evaluate with LLM-as-judge"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to aggregated results CSV file"
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs='?',
        const=TEST_MODE_RECORD_LIMIT,
        help="Run in test mode (process first N records, default: 3)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing (skip rows already in output file)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each record"
    )
    
    args = parser.parse_args()
    
    test_mode = args.test is not None
    test_limit = args.test if test_mode else None
    
    rerank_aggregated_results(
        input_path=args.input,
        test_mode=test_mode,
        test_limit=test_limit,
        resume=args.resume,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
