"""
Run Model Script - Generate analogies using a single LLM model
Supports parallel execution by running multiple instances with different models
"""

import argparse
import sys
import os
import ast
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
import json

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage1_analysis', 'mapping_generation'))

import dspy
from easy_llm_importer import LLMClient, DSPyAdapter

from config import (
    MODEL_LIST,
    SCAR_PATH,
    RESULTS_DIR,
    TEST_MODE_RECORD_LIMIT,
    ANALOGY_GENERATION_PROMPT,
    MAPPING_MODEL,
    NUM_ANALOGIES,
    get_output_path,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_gold_sources(df: pd.DataFrame, target: str) -> List[str]:
    """
    Get all gold sources for a given target from the dataset.
    
    Args:
        df: SCAR dataset DataFrame
        target: Target system name
        
    Returns:
        List of all unique gold source names for this target
    """
    target_rows = df[df['system_a'] == target]
    gold_sources = target_rows['system_b'].dropna().unique().tolist()
    return [str(gs).strip() for gs in gold_sources if str(gs).strip()]


def deduplicate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate targets by keeping one row per unique target and collecting all gold sources.
    
    Args:
        df: Original SCAR DataFrame (may have duplicate targets with different gold sources)
        
    Returns:
        DataFrame with unique targets, each with all_gold_sources column
    """
    # Group by target (system_a) and aggregate
    unique_targets = []
    
    for target in df['system_a'].unique():
        target_rows = df[df['system_a'] == target]
        
        # Get first row as base (for background, etc.)
        first_row = target_rows.iloc[0].copy()
        
        # Collect all unique gold sources for this target
        all_gold_sources = target_rows['system_b'].dropna().unique().tolist()
        all_gold_sources = [str(gs).strip() for gs in all_gold_sources if str(gs).strip()]
        
        # Collect all sub-concepts mappings (we'll use the first one for generation)
        # but store all for reference
        first_row['all_gold_sources'] = json.dumps(all_gold_sources)
        first_row['num_gold_sources'] = len(all_gold_sources)
        
        unique_targets.append(first_row)
    
    result_df = pd.DataFrame(unique_targets)
    print(f"Deduplicated {len(df)} records to {len(result_df)} unique targets")
    
    return result_df


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

def _create_analogy_signature(num_analogies: int):
    """
    Dynamically create an AnalogyGeneratorSignature with the specified number of output fields.
    This allows configuring NUM_ANALOGIES without manually editing field definitions.
    """
    # Ordinal suffixes for descriptions
    ordinals = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", 
                "Ninth", "Tenth", "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth",
                "Sixteenth", "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth",
                "Twenty-first", "Twenty-second", "Twenty-third", "Twenty-fourth", "Twenty-fifth"]
    
    # Build class attributes dict
    attrs = {
        '__doc__': f"Generate {num_analogies} source concepts that could serve as analogies for the target.",
        '__annotations__': {
            'target_concept': str,
            'sub_concepts': str,
        },
        'target_concept': dspy.InputField(desc="The unfamiliar scientific concept to explain"),
        'sub_concepts': dspy.InputField(desc="Key sub-concepts/components of the target (empty if not provided)"),
    }
    
    # Add output fields dynamically
    for i in range(1, num_analogies + 1):
        field_name = f"analogy_{i}"
        ordinal = ordinals[i-1] if i <= len(ordinals) else f"#{i}"
        attrs['__annotations__'][field_name] = str
        attrs[field_name] = dspy.OutputField(
            desc=f"{ordinal} familiar source concept as analogy (1-3 words only, no explanation or additional text)"
        )
    
    # Create and return the class
    return type('AnalogyGeneratorSignature', (dspy.Signature,), attrs)

# Create the signature class with configured number of analogies
AnalogyGeneratorSignature = _create_analogy_signature(NUM_ANALOGIES)

class SourcePropertyMappingSignature(dspy.Signature):
    """Map target sub-concepts to corresponding source sub-concepts for an analogy."""
    
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar/target concept")
    description_of_unfamiliar_concept: str = dspy.InputField(desc="Description of the target concept")
    properties_of_unfamiliar_concept: list[str] = dspy.InputField(desc="Target sub-concepts to map")
    familiar_concept: str = dspy.InputField(desc="The familiar/source analogy concept")
    mapped_source_properties: dict[str, str] = dspy.OutputField(desc="Dictionary mapping target property to source property (1-2 words each)")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_subconcepts(record: pd.Series) -> str:
    """
    Extract sub-concepts from the mappings_parsed column (system_a side).
    
    Args:
        record: A pandas Series containing the SCAR record
        
    Returns:
        Comma-separated string of sub-concepts
    """
    mappings_str = record.get('mappings_parsed', '[]')
    
    if pd.isna(mappings_str):
        return ""
    
    try:
        mappings = ast.literal_eval(mappings_str)
        # Extract the first element (system_a side) from each mapping
        subconcepts = [m[0] for m in mappings if isinstance(m, list) and len(m) >= 1]
        return ", ".join(subconcepts)
    except (ValueError, SyntaxError):
        return ""


def extract_analogies_from_result(result, with_mappings: bool = False) -> List[str]:
    """
    Extract analogies from a DSPy result.
    
    Args:
        result: DSPy prediction result
        with_mappings: Deprecated parameter, kept for backwards compatibility
        
    Returns:
        List of analogy strings (up to NUM_ANALOGIES)
    """
    analogies = []
    
    for i in range(1, NUM_ANALOGIES + 1):
        attr = f"analogy_{i}"
        if hasattr(result, attr):
            analogy = getattr(result, attr)
            if analogy:
                analogies.append(str(analogy).strip())
    
    return analogies


def generate_analogies_for_record(
    record: pd.Series,
    generator: dspy.Module,
    mode: str,
    verbose: bool = False
) -> Dict:
    """
    Generate analogies for a single SCAR record using a two-stage approach.
    
    Stage 1: Generate NUM_ANALOGIES analogies using AnalogyGeneratorSignature (same for both modes)
    Stage 2 (withsub only): For each analogy, generate sub-concept mappings
    
    Args:
        record: SCAR record as pandas Series (deduplicated, with all_gold_sources column)
        generator: DSPy generator module for Stage 1
        mode: 'targetonly' or 'withsub'
        verbose: Whether to print detailed output
        
    Returns:
        Dict with id, target, all_gold_sources, sub_concepts, generated_analogies
        For withsub mode, also includes analogy_subconcepts (mapped sub-concepts for each analogy)
    """
    target = record['system_a']
    record_id = record['id']
    target_description = record.get('system_a_background', '')
    
    # Get all gold sources (from deduplicated data)
    all_gold_sources = []
    if 'all_gold_sources' in record and pd.notna(record['all_gold_sources']):
        try:
            all_gold_sources = json.loads(record['all_gold_sources'])
        except (json.JSONDecodeError, TypeError):
            # Fallback to system_b if parsing fails
            if pd.notna(record.get('system_b')):
                all_gold_sources = [str(record['system_b']).strip()]
    else:
        # Fallback to system_b if all_gold_sources not available
        if pd.notna(record.get('system_b')):
            all_gold_sources = [str(record['system_b']).strip()]
    
    # Get sub-concepts if needed
    sub_concepts = ""
    if mode == "withsub":
        sub_concepts = extract_subconcepts(record)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Record ID: {record_id}")
        print(f"Target: {target}")
        print(f"Gold Sources ({len(all_gold_sources)}): {all_gold_sources}")
        if mode == "withsub":
            print(f"Target Sub-concepts: {sub_concepts}")
    
    try:
        # Check if generator has _instructions attribute (models without instructions support)
        has_instructions_attr = hasattr(generator, '_instructions')
        
        # =====================================================================
        # STAGE 1: Generate analogies
        # - targetonly: Only target concept is provided (sub_concepts = "")
        # - withsub: Target concept AND sub-concepts are provided
        # =====================================================================
        # Determine what sub_concepts to pass to the generator
        generator_sub_concepts = sub_concepts if mode == "withsub" else ""
        
        if verbose:
            print(f"Mode: {mode}")
            print(f"Sub-concepts passed to generator: '{generator_sub_concepts}'")
        
        if has_instructions_attr:
            # For models without instructions support, include instructions in the prompt
            instructions_text = generator._instructions
            enhanced_target = f"{instructions_text}\n\nNow, generate analogies for this target concept: {target}"
            result = generator(
                target_concept=enhanced_target,
                sub_concepts=generator_sub_concepts
            )
        else:
            # Normal call with instructions parameter (supported models)
            result = generator(
                target_concept=target,
                sub_concepts=generator_sub_concepts
            )
        
        reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
        analogies = extract_analogies_from_result(result, with_mappings=False)
        
        if verbose:
            print(f"Generated Analogies:")
            for i, a in enumerate(analogies, 1):
                print(f"  {i}. {a}")
        
        # =====================================================================
        # STAGE 2: Generate mappings (withsub mode only)
        # =====================================================================
        analogy_subconcepts = []
        
        if mode == "withsub" and sub_concepts:
            if verbose:
                print(f"\nGenerating sub-concept mappings using {MAPPING_MODEL}...")
            
            # Create a separate LM client for sub-concept mapping
            mapping_client = LLMClient()
            mapping_adapter = DSPyAdapter(mapping_client, MAPPING_MODEL)
            mapping_lm = mapping_adapter.get_dspy_lm()
            
            # Use the mapping model for sub-concept mapping
            with dspy.context(lm=mapping_lm):
                mapper = dspy.Predict(SourcePropertyMappingSignature)
                
                # Parse target sub-concepts into a list
                target_props = [p.strip() for p in sub_concepts.split(",")]
                
                for analogy in analogies:
                    try:
                        mapping_result = mapper(
                            unfamiliar_concept=target,
                            description_of_unfamiliar_concept=target_description if target_description else "",
                            properties_of_unfamiliar_concept=target_props,
                            familiar_concept=analogy
                        )
                        
                        # Convert dict to comma-separated string (values in order of target props)
                        mapped_props = mapping_result.mapped_source_properties
                        if isinstance(mapped_props, dict):
                            # Maintain order based on target_props
                            ordered_values = [mapped_props.get(prop, "") for prop in target_props]
                            analogy_subconcepts.append(", ".join(ordered_values))
                        else:
                            analogy_subconcepts.append("")
                            
                    except Exception as mapping_error:
                        if verbose:
                            print(f"    Warning: Mapping failed for '{analogy}': {mapping_error}")
                        analogy_subconcepts.append("")
            
            if verbose:
                print(f"\nAnalogies with Mappings:")
                for i, (a, sc) in enumerate(zip(analogies, analogy_subconcepts), 1):
                    print(f"  {i}. {a}")
                    print(f"     └─ Mapped sub-concepts: {sc}")
        
        return {
            'id': record_id,
            'target': target,
            'all_gold_sources': json.dumps(all_gold_sources),
            'num_gold_sources': len(all_gold_sources),
            'sub_concepts': sub_concepts,
            'generated_analogies': json.dumps(analogies),
            'analogy_subconcepts': json.dumps(analogy_subconcepts),
            'reasoning': reasoning,
            'status': 'success',
        }
        
    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        
        return {
            'id': record_id,
            'target': target,
            'all_gold_sources': json.dumps(all_gold_sources),
            'num_gold_sources': len(all_gold_sources),
            'sub_concepts': sub_concepts,
            'generated_analogies': json.dumps([]),
            'analogy_subconcepts': json.dumps([]),
            'reasoning': "",
            'status': f'error: {str(e)}'
        }


def run_model(
    model_name: str,
    mode: str,
    test_mode: bool = False,
    verbose: bool = False
):
    """
    Run analogy generation for all records using a specific model.
    
    Args:
        model_name: Name of the model from easy_llm_importer
        mode: 'targetonly' or 'withsub'
        test_mode: If True, only process TEST_MODE_RECORD_LIMIT records
        verbose: Whether to print detailed output
    """
    print("=" * 70)
    print(f"Running Model: {model_name}")
    print(f"Mode: {mode}")
    print(f"Test Mode: {test_mode}")
    print("=" * 70)
    
    # Load SCAR dataset
    print(f"\nLoading SCAR dataset from {SCAR_PATH}")
    df = pd.read_csv(SCAR_PATH)
    print(f"Loaded {len(df)} records")
    
    # Deduplicate targets (collect all gold sources per unique target)
    df = deduplicate_targets(df)
    
    # Limit records in test mode
    if test_mode:
        df = df.head(TEST_MODE_RECORD_LIMIT)
        print(f"Test mode: Using first {len(df)} unique targets")
        verbose = True  # Force verbose in test mode
    
    # Initialize LLM client and DSPy
    print(f"\nInitializing LLM client for {model_name}...")
    client = LLMClient()
    adapter = DSPyAdapter(client, model_name)
    lm = adapter.get_dspy_lm()
    dspy.configure(lm=lm)
    
    # Check if model supports instructions parameter
    # Some models like gpt-4.1-mini and gpt-4.1-nano don't support it
    models_without_instructions = ["gpt-4.1-mini", "gpt-4.1-nano"]
    supports_instructions = model_name.lower() not in [m.lower() for m in models_without_instructions]
    
    # Create generator - same signature for both modes (two-stage approach)
    # Stage 1: Generate analogies using AnalogyGeneratorSignature
    # Stage 2 (withsub only): Map sub-concepts using SourcePropertyMappingSignature
    if supports_instructions:
        generator = dspy.ChainOfThought(
            AnalogyGeneratorSignature,
            instructions=ANALOGY_GENERATION_PROMPT
        )
    else:
        # For models without instructions support, use Predict
        generator = dspy.Predict(
            AnalogyGeneratorSignature
        )
        # Store instructions for later use in prompt formatting
        generator._instructions = ANALOGY_GENERATION_PROMPT
    
    # Process all records
    results = []
    print(f"\nGenerating analogies for {len(df)} records...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {model_name}"):
        result = generate_analogies_for_record(row, generator, mode, verbose)
        results.append(result)
    
    # Save results
    output_path = get_output_path(model_name, mode, is_eval=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"Total unique targets: {len(results_df)}")
    print(f"Total gold sources covered: {results_df['num_gold_sources'].sum()}")
    print(f"Successful: {len(results_df[results_df['status'] == 'success'])}")
    print(f"Errors: {len(results_df[results_df['status'] != 'success'])}")
    print("=" * 70)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run LLM analogy generation")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=MODEL_LIST,
        help="Model name to use for generation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["targetonly", "withsub"],
        help="Generation mode: targetonly or withsub (with sub-concepts)"
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
    
    run_model(
        model_name=args.model,
        mode=args.mode,
        test_mode=args.test,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
