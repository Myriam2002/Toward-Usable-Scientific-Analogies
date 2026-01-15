"""
Retry Failed Records: Find and re-run failed LLM calls in output JSON files.
Scans all JSON files in parallel_runners/outputs/ for records with success=false,
retries them with the appropriate model and experiment type, then updates the files.
"""

import os
import sys
import json
import time
import random
import ast
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add parent directories to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import dspy
from easy_llm_importer import create_client, DSPyAdapter

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'parallel_runners', 'outputs')
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'SCAR_cleaned_manually.csv')
MAX_RETRIES = 3

# Credit error keywords to detect
CREDIT_ERROR_KEYWORDS = [
    'requires more credits',
    'insufficient credits',
    'credit limit',
    'rate limit',
    'quota exceeded',
    'afford'
]


# ============================================================================
# DSPY SIGNATURES (same as in runner scripts)
# ============================================================================

class PropertyMatching(dspy.Signature):
    """Given both unfamiliar and familiar concept properties, create the correct mappings between them."""
    
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar concept for student to learn")
    properties_of_unfamiliar_concept: list[str] = dspy.InputField(desc="List of key properties that characterize the unfamiliar concept. Each property is 1-2 words maximum.")
    familiar_concept: str = dspy.InputField(desc="The familiar concept used to create the analogy")
    properties_of_familiar_concept: list[str] = dspy.InputField(desc="List of key properties that characterize the familiar concept. Each property is 1-2 words maximum.")
    mapped_source_properties: dict[str, str] = dspy.OutputField(desc="Dictionary mapping each unfamiliar concept property to corresponding familiar concept property")


class PropertyMatchingWithDescription(dspy.Signature):
    """Given both unfamiliar and familiar concept properties with descriptions, create accurate mappings."""
    
    unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar concept for student to learn")
    description_of_unfamiliar_concept: str = dspy.InputField(desc="Detailed description or context about the unfamiliar concept")
    properties_of_unfamiliar_concept: list[str] = dspy.InputField(desc="List of key properties that characterize the unfamiliar concept. Each property is 1-2 words maximum.")
    familiar_concept: str = dspy.InputField(desc="The familiar concept used to create the analogy")
    description_of_familiar_concept: str = dspy.InputField(desc="Detailed description or context about the familiar concept")
    properties_of_familiar_concept: list[str] = dspy.InputField(desc="List of key properties that characterize the familiar concept. Each property is 1-2 words maximum.")
    mapped_source_properties: dict[str, str] = dspy.OutputField(desc="Dictionary mapping each unfamiliar concept property to corresponding familiar concept property")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_text(text):
    """Normalize text for comparison (lowercase, strip whitespace)"""
    if text is None:
        return ""
    return str(text).lower().strip()


def is_credit_error(error_message):
    """Check if an error is related to credits/rate limits"""
    if not error_message:
        return False
    error_lower = error_message.lower()
    return any(keyword in error_lower for keyword in CREDIT_ERROR_KEYWORDS)


def calculate_mapping_accuracy(ground_truth_mappings, predicted_mappings):
    """Calculate accuracy metrics for predicted vs ground truth mappings"""
    if not ground_truth_mappings or not predicted_mappings:
        return {
            'system_accuracy': False,
            'correct_mappings': 0,
            'total_mappings': len(ground_truth_mappings) if ground_truth_mappings else 0,
            'concept_mapping_accuracy': 0.0
        }
    
    total_mappings = len(ground_truth_mappings)
    correct_mappings = 0
    
    for unfam_concept, gt_familiar_concept in ground_truth_mappings.items():
        unfam_normalized = normalize_text(unfam_concept)
        
        predicted_familiar = None
        for pred_key, pred_value in predicted_mappings.items():
            if normalize_text(pred_key) == unfam_normalized:
                predicted_familiar = pred_value
                break
        
        if predicted_familiar is not None:
            gt_normalized = normalize_text(gt_familiar_concept)
            pred_normalized = normalize_text(predicted_familiar)
            
            if gt_normalized == pred_normalized:
                correct_mappings += 1
    
    concept_mapping_accuracy = correct_mappings / total_mappings if total_mappings > 0 else 0.0
    system_accuracy = (correct_mappings == total_mappings)
    
    return {
        'system_accuracy': system_accuracy,
        'correct_mappings': correct_mappings,
        'total_mappings': total_mappings,
        'concept_mapping_accuracy': concept_mapping_accuracy
    }


def extract_model_and_experiment(filename):
    """Extract model name and experiment type from filename.
    
    Examples:
        2c_deepseek-r1.json -> ('deepseek-r1', '2c')
        2d_gpt-4.1-mini.json -> ('gpt-4.1-mini', '2d')
    """
    basename = os.path.basename(filename)
    name_without_ext = basename.replace('.json', '')
    
    if name_without_ext.startswith('2c_'):
        return name_without_ext[3:], '2c'
    elif name_without_ext.startswith('2d_'):
        return name_without_ext[3:], '2d'
    else:
        return None, None


def load_scar_dataset():
    """Load and prepare the SCAR dataset for description lookups."""
    df_scar = pd.read_csv(DATA_PATH)
    
    # Create a lookup dictionary by (unfamiliar_concept, familiar_concept)
    lookup = {}
    for _, row in df_scar.iterrows():
        key = (row['system_a'].strip(), row['system_b'].strip())
        lookup[key] = {
            'description_unfamiliar': row.get('system_a_background', ''),
            'description_familiar': row.get('system_b_background', '')
        }
    
    return lookup


def run_property_matching_no_desc(record):
    """Run PropertyMatching (no description) on a record with SHUFFLED properties"""
    predictor = dspy.ChainOfThought(PropertyMatching)
    
    # Shuffle unfamiliar property list (deterministic per row for reproducibility)
    random.seed(record['id'])
    unfamiliar_props_shuffled = record['ground_truth_properties_unfamiliar'].copy()
    familiar_props_shuffled = record['ground_truth_properties_familiar'].copy()
    random.shuffle(unfamiliar_props_shuffled)
    
    result = predictor(
        unfamiliar_concept=record['unfamiliar_concept'],
        properties_of_unfamiliar_concept=unfamiliar_props_shuffled,
        familiar_concept=record['familiar_concept'],
        properties_of_familiar_concept=familiar_props_shuffled
    )
    reasoning = result.reasoning
    return result.mapped_source_properties, reasoning


def run_property_matching_with_desc(record, description_unfamiliar, description_familiar):
    """Run PropertyMatchingWithDescription on a record with SHUFFLED properties"""
    predictor = dspy.ChainOfThought(PropertyMatchingWithDescription)
    
    # Shuffle unfamiliar property list (deterministic per row for reproducibility)
    random.seed(record['id'])
    unfamiliar_props_shuffled = record['ground_truth_properties_unfamiliar'].copy()
    familiar_props_shuffled = record['ground_truth_properties_familiar'].copy()
    random.shuffle(unfamiliar_props_shuffled)
    
    result = predictor(
        unfamiliar_concept=record['unfamiliar_concept'],
        description_of_unfamiliar_concept=description_unfamiliar,
        properties_of_unfamiliar_concept=unfamiliar_props_shuffled,
        familiar_concept=record['familiar_concept'],
        description_of_familiar_concept=description_familiar,
        properties_of_familiar_concept=familiar_props_shuffled
    )
    reasoning = result.reasoning    
    return result.mapped_source_properties, reasoning


def retry_record(record, experiment_type, model_name, scar_lookup):
    """Retry a single failed record.
    
    Returns:
        tuple: (success, updated_record)
    """
    start_time = time.time()
    
    try:
        if experiment_type == '2c':
            result, reasoning = run_property_matching_no_desc(record)
        else:  # 2d
            # Look up descriptions
            key = (record['unfamiliar_concept'], record['familiar_concept'])
            descriptions = scar_lookup.get(key, {})
            description_unfamiliar = descriptions.get('description_unfamiliar', '')
            description_familiar = descriptions.get('description_familiar', '')
            
            result, reasoning = run_property_matching_with_desc(
                record, description_unfamiliar, description_familiar
            )
        
        duration = time.time() - start_time
        
        # Calculate accuracy
        ground_truth_mappings = record['ground_truth_mappings']
        accuracy_metrics = calculate_mapping_accuracy(ground_truth_mappings, result)
        
        # Update record with success
        record['success'] = True
        record['error'] = None
        record['duration_seconds'] = duration
        record['timestamp'] = datetime.now().isoformat()
        record['predicted_mappings'] = result
        record['reasoning'] = reasoning
        record['system_accuracy'] = accuracy_metrics['system_accuracy']
        record['correct_mappings'] = accuracy_metrics['correct_mappings']
        record['total_mappings'] = accuracy_metrics['total_mappings']
        record['concept_mapping_accuracy'] = accuracy_metrics['concept_mapping_accuracy']
        
        return True, record
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        
        # Check if it's a credit error
        if is_credit_error(error_msg):
            print(f"\n  ⚠️  CREDIT ERROR for {model_name}: {error_msg[:100]}...")
            record['error'] = error_msg
            return False, record  # Return False to stop retrying this model
        
        # Update record with new error
        record['error'] = error_msg
        record['duration_seconds'] = duration
        record['timestamp'] = datetime.now().isoformat()
        
        return None, record  # None means retry again


def configure_dspy_for_model(model_name, client):
    """Configure DSPy settings for a specific model."""
    adapter = DSPyAdapter(client, model_name=model_name)
    lm = adapter.get_dspy_lm()
    dspy.settings.configure(lm=lm)


def process_json_file(filepath, client, scar_lookup):
    """Process a single JSON file and retry all failed records.
    
    Returns:
        tuple: (total_failed, retried_success, still_failed, credit_errors)
    """
    model_name, experiment_type = extract_model_and_experiment(filepath)
    
    if model_name is None:
        print(f"  Skipping {filepath} - unknown format")
        return 0, 0, 0, 0
    
    # Load the results
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Find failed records
    failed_indices = [i for i, r in enumerate(results) if not r.get('success', True)]
    
    if not failed_indices:
        return 0, 0, 0, 0
    
    print(f"\n  Processing: {os.path.basename(filepath)}")
    print(f"  Model: {model_name}, Experiment: {experiment_type}")
    print(f"  Found {len(failed_indices)} failed records to retry")
    
    # Configure DSPy for this model
    configure_dspy_for_model(model_name, client)
    
    retried_success = 0
    still_failed = 0
    credit_errors = 0
    credit_error_hit = False
    
    for idx in tqdm(failed_indices, desc=f"  Retrying {model_name}"):
        if credit_error_hit:
            # Skip remaining records for this model if we hit a credit error
            still_failed += 1
            continue
        
        record = results[idx]
        
        # Try up to MAX_RETRIES times
        for attempt in range(MAX_RETRIES):
            success, updated_record = retry_record(
                record, experiment_type, model_name, scar_lookup
            )
            results[idx] = updated_record
            
            if success is True:
                retried_success += 1
                break
            elif success is False:
                # Credit error - stop retrying this model
                credit_errors += 1
                credit_error_hit = True
                still_failed += 1
                break
            # If success is None, continue retrying
        else:
            # All retries exhausted
            still_failed += 1
    
    # Save the updated results
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    return len(failed_indices), retried_success, still_failed, credit_errors


def main():
    """Main function to retry all failed records."""
    print("=" * 70)
    print("RETRY FAILED RECORDS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max retries per record: {MAX_RETRIES}")
    print("=" * 70)
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"\nERROR: Output directory does not exist: {OUTPUT_DIR}")
        return False
    
    # Load SCAR dataset for descriptions
    print("\nLoading SCAR dataset for description lookups...")
    scar_lookup = load_scar_dataset()
    print(f"Loaded {len(scar_lookup)} concept pairs")
    
    # Create LLM client
    print("\nInitializing LLM client...")
    client = create_client()
    
    # Find all JSON files
    json_files = sorted([
        os.path.join(OUTPUT_DIR, f) 
        for f in os.listdir(OUTPUT_DIR) 
        if f.endswith('.json')
    ])
    
    print(f"\nFound {len(json_files)} JSON files to scan")
    
    # Process each file
    total_stats = {
        'files_with_failures': 0,
        'total_failed': 0,
        'retried_success': 0,
        'still_failed': 0,
        'credit_errors': 0
    }
    
    for filepath in json_files:
        total_failed, retried_success, still_failed, credit_errors = process_json_file(
            filepath, client, scar_lookup
        )
        
        if total_failed > 0:
            total_stats['files_with_failures'] += 1
            total_stats['total_failed'] += total_failed
            total_stats['retried_success'] += retried_success
            total_stats['still_failed'] += still_failed
            total_stats['credit_errors'] += credit_errors
            
            print(f"  Results: {retried_success}/{total_failed} fixed, "
                  f"{still_failed} still failed, {credit_errors} credit errors")
    
    # Summary
    print("\n" + "=" * 70)
    print("RETRY SUMMARY")
    print("=" * 70)
    print(f"Files with failures: {total_stats['files_with_failures']}")
    print(f"Total failed records: {total_stats['total_failed']}")
    print(f"Successfully retried: {total_stats['retried_success']}")
    print(f"Still failed: {total_stats['still_failed']}")
    print(f"Credit errors: {total_stats['credit_errors']}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if total_stats['credit_errors'] > 0:
        print("\n⚠️  Some records failed due to credit limits. Add more credits and re-run.")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        # Run aggregation
        print("\n" + "=" * 70)
        print("RUNNING AGGREGATION")
        print("=" * 70)
        
        import subprocess
        aggregate_script = os.path.join(SCRIPT_DIR, 'aggregate_results.py')
        result = subprocess.run([sys.executable, aggregate_script], cwd=SCRIPT_DIR)
        
        if result.returncode == 0:
            print("\n✓ Aggregation completed successfully")
        else:
            print("\n✗ Aggregation failed")
    
    sys.exit(0 if success else 1)

