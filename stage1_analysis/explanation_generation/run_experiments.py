"""
Explanation Generation Evaluation - Experiment Runner

This script evaluates whether models can create explanations of why two systems are analogies.
Testing different input conditions:
1. None: unfamiliar concept + familiar concept only
2. None + Description: includes background descriptions
3. Unpaired Properties: includes property lists
4. Unpaired Properties + Description: includes properties and descriptions
5. Paired Properties: includes pre-mapped property pairs
6. Paired Properties + Description: includes mapped pairs and descriptions
"""

import os
import sys
import json
import ast
import argparse
import time
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import dspy

# Setup paths and imports
load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.abspath(os.path.join(current_dir, '..', 'mapping_generation'))
sys.path.append(mapping_path)

from easy_llm_importer import create_client, list_available_models, DSPyAdapter
from explanation_evaluation import ExplanationEvaluator

# ============================================================================
# CONFIGURATION
# ============================================================================
# Note: These can now be overridden via command-line arguments.
# Run with --help to see all options.

# Available experiment settings
EXPERIMENT_SETTINGS = [
    'none',
    'none_description',
    'unpaired_properties',
    'unpaired_properties_description',
    'paired_properties',
    'paired_properties_description'
]

# API call configuration
MAX_RETRIES = 3  # Maximum number of retry attempts
RETRY_DELAY = 5  # Initial delay in seconds before retry
RETRY_BACKOFF = 2  # Exponential backoff multiplier
REQUEST_DELAY = 1.0  # Delay between successful API calls (in seconds)

# ============================================================================
# DSPy SIGNATURE CLASSES
# ============================================================================

class ExplanationGeneration_None(dspy.Signature):
    """Extract key properties and attributes from an unfamiliar concept."""
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar concept for student to learn")
    familiar_concept: str = dspy.InputField(desc="The familiar concept used to create the analogy")
    Explanation: str = dspy.OutputField(desc="The explanation of how the unfamiliar concept and the familiar concept are analogies according to the properties of each concept")


class ExplanationGeneration_None_description(dspy.Signature):
    """Extract key properties and attributes from an unfamiliar concept."""
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar concept for student to learn")
    description_of_unfamiliar_concept: str = dspy.InputField(desc="The description of the unfamiliar concept")
    familiar_concept: str = dspy.InputField(desc="The familiar concept used to create the analogy")
    description_of_familiar_concept: str = dspy.InputField(desc="The description of the familiar concept")
    Explanation: str = dspy.OutputField(desc="The explanation of how the unfamiliar concept and the familiar concept are analogies according to the properties of each concept")


class ExplanationGeneration_UnpairedProperties(dspy.Signature):
    """Extract key properties and attributes from an unfamiliar concept."""
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar concept for student to learn")
    properties_of_unfamiliar_concept: list[str] = dspy.InputField(desc="List of key properties that characterize the unfamiliar concept. Each property is 1-2 words maximum.")
    familiar_concept: str = dspy.InputField(desc="The familiar concept used to create the analogy")
    properties_of_familiar_concept: list[str] = dspy.InputField(desc="List of key properties that characterize the familiar concept. Each property is 1-2 words maximum.")
    Explanation: list[str] = dspy.OutputField(desc="The explanation of how the unfamiliar concept and the familiar concept are analogies according to each of the properties of each concept. Pair them first and then explain the analogy for each pair.")


class ExplanationGeneration_UnpairedProperties_description(dspy.Signature):
    """Extract key properties and attributes from an unfamiliar concept."""
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar concept for student to learn")
    description_of_unfamiliar_concept: str = dspy.InputField(desc="The description of the unfamiliar concept")
    properties_of_unfamiliar_concept: list[str] = dspy.InputField(desc="List of key properties that characterize the unfamiliar concept. Each property is 1-2 words maximum.")
    familiar_concept: str = dspy.InputField(desc="The familiar concept used to create the analogy")
    description_of_familiar_concept: str = dspy.InputField(desc="The description of the familiar concept")
    properties_of_familiar_concept: list[str] = dspy.InputField(desc="List of key properties that characterize the familiar concept. Each property is 1-2 words maximum.")
    Explanation: list[str] = dspy.OutputField(desc="The explanation of how the unfamiliar concept and the familiar concept are analogies according to each of the properties of each concept. Pair them first and then explain the analogy for each pair.")


class ExplanationGeneration_PairedProperties(dspy.Signature):
    """Extract key properties and attributes from an unfamiliar concept."""
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar concept for student to learn")
    familiar_concept: str = dspy.InputField(desc="The familiar concept used to create the analogy")
    paired_properties: list[list[str]] = dspy.InputField(desc="Dictionary mapping each unfamiliar concept property to corresponding familiar concept property (1-2 words each)")
    Explanation: list[str] = dspy.OutputField(desc="The explanation of how the unfamiliar concept and the familiar concept are analogies according to each of the paired properties")


class ExplanationGeneration_PairedProperties_description(dspy.Signature):
    """Extract key properties and attributes from an unfamiliar concept."""
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar concept for student to learn")
    description_of_unfamiliar_concept: str = dspy.InputField(desc="The description of the unfamiliar concept")
    familiar_concept: str = dspy.InputField(desc="The familiar concept used to create the analogy")
    description_of_familiar_concept: str = dspy.InputField(desc="The description of the familiar concept")
    paired_properties: list[list[str]] = dspy.InputField(desc="Dictionary mapping each unfamiliar concept property to corresponding familiar concept property (1-2 words each)")
    Explanation: list[str] = dspy.OutputField(desc="The explanation of how the unfamiliar concept and the familiar concept are analogies according to each of the paired properties")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_explanation_generation_none(row):
    """Run explanation generation with only concept names."""
    extractor = dspy.ChainOfThought(ExplanationGeneration_None)
    result = extractor(unfamiliar_concept=row['unfamiliar_concept'], familiar_concept=row['familiar_concept'])
    return result.Explanation, result.reasoning


def run_explanation_generation_none_description(row):
    """Run explanation generation with concept names and descriptions."""
    extractor = dspy.ChainOfThought(ExplanationGeneration_None_description)
    result = extractor(
        unfamiliar_concept=row['unfamiliar_concept'],
        description_of_unfamiliar_concept=row['description_unfamiliar'],
        familiar_concept=row['familiar_concept'],
        description_of_familiar_concept=row['description_familiar']
    )
    return result.Explanation, result.reasoning


def run_explanation_generation_unpaired_properties(row):
    """Run explanation generation with concept names and unpaired property lists."""
    extractor = dspy.ChainOfThought(ExplanationGeneration_UnpairedProperties)
    result = extractor(
        unfamiliar_concept=row['unfamiliar_concept'],
        properties_of_unfamiliar_concept=row['properties_unfamiliar'],
        familiar_concept=row['familiar_concept'],
        properties_of_familiar_concept=row['properties_familiar']
    )
    return result.Explanation, result.reasoning


def run_explanation_generation_unpaired_properties_description(row):
    """Run explanation generation with concept names, descriptions, and unpaired property lists."""
    extractor = dspy.ChainOfThought(ExplanationGeneration_UnpairedProperties_description)
    result = extractor(
        unfamiliar_concept=row['unfamiliar_concept'],
        description_of_unfamiliar_concept=row['description_unfamiliar'],
        properties_of_unfamiliar_concept=row['properties_unfamiliar'],
        familiar_concept=row['familiar_concept'],
        description_of_familiar_concept=row['description_familiar'],
        properties_of_familiar_concept=row['properties_familiar']
    )
    return result.Explanation, result.reasoning


def run_explanation_generation_paired_properties(row):
    """Run explanation generation with concept names and paired property mappings."""
    extractor = dspy.ChainOfThought(ExplanationGeneration_PairedProperties)
    result = extractor(
        unfamiliar_concept=row['unfamiliar_concept'],
        familiar_concept=row['familiar_concept'],
        paired_properties=row['mappings_list']
    )
    return result.Explanation, result.reasoning


def run_explanation_generation_paired_properties_description(row):
    """Run explanation generation with concept names, descriptions, and paired property mappings."""
    extractor = dspy.ChainOfThought(ExplanationGeneration_PairedProperties_description)
    result = extractor(
        unfamiliar_concept=row['unfamiliar_concept'],
        description_of_unfamiliar_concept=row['description_unfamiliar'],
        familiar_concept=row['familiar_concept'],
        description_of_familiar_concept=row['description_familiar'],
        paired_properties=row['mappings_list']
    )
    return result.Explanation, result.reasoning


# Mapping of setting names to functions
EXPERIMENT_FUNCTIONS = {
    'none': run_explanation_generation_none,
    'none_description': run_explanation_generation_none_description,
    'unpaired_properties': run_explanation_generation_unpaired_properties,
    'unpaired_properties_description': run_explanation_generation_unpaired_properties_description,
    'paired_properties': run_explanation_generation_paired_properties,
    'paired_properties_description': run_explanation_generation_paired_properties_description
}


def load_and_prepare_data(data_path):
    """Load and prepare the SCAR dataset."""
    print(f"Loading data from {data_path}...")
    df_scar = pd.read_csv(data_path)
    
    # Parse mappings and extract properties
    df_scar['mappings_list'] = df_scar['mappings_parsed'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and x else []
    )
    df_scar['properties_unfamiliar'] = df_scar['mappings_list'].apply(
        lambda x: [pair[0] for pair in x] if x else []
    )
    df_scar['properties_familiar'] = df_scar['mappings_list'].apply(
        lambda x: [pair[1] for pair in x] if x else []
    )
    
    # Parse explanations
    df_scar['explanation_list'] = df_scar['explanation_parsed'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and x else []
    )
    
    # Clean concept names
    df_scar['unfamiliar_concept'] = df_scar['system_a'].str.strip()
    df_scar['familiar_concept'] = df_scar['system_b'].str.strip()
    
    # Use background descriptions
    df_scar['description_unfamiliar'] = df_scar['system_a_background']
    df_scar['description_familiar'] = df_scar['system_b_background']
    
    print(f"✅ Loaded and prepared {len(df_scar)} rows")
    return df_scar


def evaluate_single_explanation(evaluator, golden_explanation, predicted_explanation):
    """Evaluate a single explanation pair."""
    return evaluator.evaluate_explanation(golden_explanation, predicted_explanation)


def call_with_retry(func, *args, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY, 
                    backoff_multiplier=RETRY_BACKOFF, **kwargs):
    """
    Call a function with retry logic and exponential backoff.
    
    Args:
        func: Function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay before retry (in seconds)
        backoff_multiplier: Multiplier for exponential backoff
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result of the function call
    
    Raises:
        Exception: The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Check if this is a retryable error
            retryable_errors = ['rate limit', 'ratelimit', 'model busy', 'timeout', 
                               'connection', 'temporarily unavailable', '429', '503', '502']
            is_retryable = any(err in error_msg for err in retryable_errors)
            
            if not is_retryable:
                # Not a retryable error, raise immediately
                raise
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (backoff_multiplier ** attempt)
                print(f"⚠️  Retryable error: {str(e)[:100]}")
                print(f"   Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ All {max_retries} retry attempts failed")
                raise last_exception


def run_experiment_for_setting(client, models, setting, df_subset, evaluator, output_dir, 
                               max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY, 
                               request_delay=REQUEST_DELAY):
    """Run experiments for a specific setting across all models."""
    print(f"\n{'='*80}")
    print(f"Running experiments for setting: {setting}")
    print(f"{'='*80}\n")
    
    experiment_func = EXPERIMENT_FUNCTIONS[setting]
    json_results = []
    
    for model in tqdm(models, desc=f"Models ({setting})"):
        print(f"\n🔧 Configuring model: {model}")
        
        # Configure DSPy with current model
        try:
            adapter = DSPyAdapter(client, model_name=model)
            lm = adapter.get_dspy_lm()
            dspy.settings.configure(lm=lm)
        except Exception as e:
            print(f"❌ Failed to configure model {model}: {str(e)}")
            continue
        
        # Run experiment on each row
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"{model} rows"):
            try:
                # Call experiment function with retry logic
                result, reasoning = call_with_retry(
                    experiment_func, row, 
                    max_retries=max_retries, 
                    retry_delay=retry_delay
                )
                
                # Handle list vs string results
                if isinstance(result, list):
                    result_str = " ".join(result)
                else:
                    result_str = result
                
                # Prepare golden explanation
                golden_explanation = ", ".join(row['explanation_list'])
                
                # Evaluate
                evaluation_result = evaluate_single_explanation(evaluator, golden_explanation, result_str)
                
                json_results.append({
                    "model": model,
                    "setting": setting,
                    "row_index": idx,
                    "error": None,
                    "unfamiliar_concept": row['unfamiliar_concept'],
                    "familiar_concept": row['familiar_concept'],
                    "description_unfamiliar": row['description_unfamiliar'],
                    "description_familiar": row['description_familiar'],
                    "properties_unfamiliar": row['properties_unfamiliar'],
                    "properties_familiar": row['properties_familiar'],
                    "explanation_list": row['explanation_list'],
                    "result": result_str,
                    "reasoning": reasoning,
                    "sbert_similarity": evaluation_result['sbert_similarity'],
                    "evaluation_type": evaluation_result['evaluation_type']
                })
                
                # Add delay between successful API calls to avoid rate limits
                time.sleep(request_delay)
                
            except Exception as e:
                print(f"❌ Error processing row {idx} with {model}: {str(e)}")
                json_results.append({
                    "model": model,
                    "setting": setting,
                    "row_index": idx,
                    "error": str(e),
                    "unfamiliar_concept": row['unfamiliar_concept'],
                    "familiar_concept": row['familiar_concept'],
                    "description_unfamiliar": row.get('description_unfamiliar', ''),
                    "description_familiar": row.get('description_familiar', ''),
                    "properties_unfamiliar": row.get('properties_unfamiliar', []),
                    "properties_familiar": row.get('properties_familiar', []),
                    "explanation_list": row.get('explanation_list', []),
                    "result": "",
                    "reasoning": "",
                    "sbert_similarity": None,
                    "evaluation_type": ""
                })
                continue
        
        # Save individual model checkpoint
        model_results = [r for r in json_results if r['model'] == model]
        if model_results:
            checkpoint_df = pd.DataFrame(model_results)
            checkpoint_df.to_csv(
                os.path.join(output_dir, f'{setting}_{model}_checkpoint.csv'),
                index=False
            )
            with open(os.path.join(output_dir, f'{setting}_{model}_checkpoint.json'), 'w') as f:
                json.dump(model_results, f, indent=2)
    
    # Save combined results for this setting
    if json_results:
        combined_df = pd.DataFrame(json_results)
        combined_df.to_csv(
            os.path.join(output_dir, f'{setting}_ALL_MODELS_combined.csv'),
            index=False
        )
        with open(os.path.join(output_dir, f'{setting}_ALL_MODELS_combined.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ Completed setting '{setting}': {len(json_results)} total results")
        
        # Print summary
        print(f"\n📊 Summary for {setting}:")
        summary = combined_df.groupby('model').agg({
            'row_index': 'count',
            'error': lambda x: sum(pd.notna(x) & (x != 'None') & (x != ''))
        }).rename(columns={'row_index': 'total_rows', 'error': 'errors'})
        print(summary)
    
    return json_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run explanation generation experiments across all models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all settings in test mode (5 rows, all models)
  python run_experiments.py --test-mode
  
  # Run specific setting with all models (full dataset)
  python run_experiments.py --setting none
  
  # Run specific model with all settings (full dataset)
  python run_experiments.py --model gpt-4o-mini
  
  # Run specific setting with specific model
  python run_experiments.py --setting unpaired_properties --model gpt-4o-mini
  
  # Run multiple models in parallel (different terminals, same setting)
  python run_experiments.py --setting unpaired_properties --model gpt-4o-mini
  python run_experiments.py --setting unpaired_properties --model claude-3-5-sonnet-20241022
  python run_experiments.py --setting unpaired_properties --model gemini-1.5-flash
  
  # Run with custom test size
  python run_experiments.py --test-mode --test-rows 10 --setting paired_properties

Available settings:
  - none: Only concept names
  - none_description: Concept names + descriptions
  - unpaired_properties: Concept names + property lists
  - unpaired_properties_description: Concept names + properties + descriptions
  - paired_properties: Concept names + paired property mappings
  - paired_properties_description: Concept names + paired mappings + descriptions
        """
    )
    
    parser.add_argument(
        '--setting',
        type=str,
        choices=EXPERIMENT_SETTINGS + ['all'],
        default='all',
        help='Experiment setting to run (default: all). Use "all" to run all settings.'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with limited rows'
    )
    parser.add_argument(
        '--test-rows',
        type=int,
        default=5,
        help='Number of rows for test mode (default: 5)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../../data/SCAR_cleaned_manually.csv',
        help='Path to data file (default: ../../data/SCAR_cleaned_manually.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints/explanation_generation',
        help='Output directory for results (default: checkpoints/explanation_generation)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=MAX_RETRIES,
        help=f'Maximum number of retry attempts for rate limits (default: {MAX_RETRIES})'
    )
    parser.add_argument(
        '--retry-delay',
        type=float,
        default=RETRY_DELAY,
        help=f'Initial delay in seconds before retry (default: {RETRY_DELAY})'
    )
    parser.add_argument(
        '--request-delay',
        type=float,
        default=REQUEST_DELAY,
        help=f'Delay in seconds between successful API calls (default: {REQUEST_DELAY})'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        help='Specific model to run (default: all). Use model name to run single model, or "all" to run all models.'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_args()
    
    print("="*80)
    print("EXPLANATION GENERATION EVALUATION - EXPERIMENT RUNNER")
    print("="*80)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory: {output_dir}")
    
    # Initialize client and get models
    print("\n🔧 Initializing LLM client...")
    client = create_client()
    all_models = list_available_models()
    
    # Filter models based on argument
    if args.model == 'all':
        models = all_models
        print(f"✅ Running ALL models: {len(models)}")
        for model in models:
            print(f"  • {model}")
    else:
        if args.model not in all_models:
            print(f"❌ Error: Model '{args.model}' not found in available models.")
            print(f"Available models: {', '.join(all_models)}")
            sys.exit(1)
        models = [args.model]
        print(f"✅ Running SINGLE model: {args.model}")
    
    # Load and prepare data
    df_scar = load_and_prepare_data(args.data_path)
    
    # Apply test mode if enabled
    if args.test_mode:
        df_subset = df_scar[:args.test_rows]
        print(f"⚠️  TEST MODE: Running on {args.test_rows} rows only")
    else:
        df_subset = df_scar
        print(f"🚀 FULL MODE: Running on all {len(df_scar)} rows")
    
    # Initialize evaluator
    print("\n🔧 Initializing SBERT evaluator...")
    evaluator = ExplanationEvaluator()
    
    # Display retry/delay configuration
    print("\n⚙️  API Call Configuration:")
    print(f"  • Max Retries: {args.max_retries}")
    print(f"  • Retry Delay: {args.retry_delay}s (exponential backoff)")
    print(f"  • Request Delay: {args.request_delay}s (between calls)")
    
    # Determine which settings to run
    if args.setting == 'all':
        settings_to_run = EXPERIMENT_SETTINGS
        print(f"\n✅ Running ALL settings: {len(settings_to_run)}")
    else:
        settings_to_run = [args.setting]
        print(f"\n✅ Running SINGLE setting: {args.setting}")
    
    # Run experiments for each setting
    all_results = {}
    for setting in settings_to_run:
        try:
            results = run_experiment_for_setting(
                client, models, setting, df_subset, evaluator, output_dir,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                request_delay=args.request_delay
            )
            all_results[setting] = results
        except Exception as e:
            print(f"❌ Failed to run experiments for setting '{setting}': {str(e)}")
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETION SUMMARY")
    print("="*80)
    for setting, results in all_results.items():
        print(f"  • {setting}: {len(results)} results")
    print(f"\n✅ All experiments completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

