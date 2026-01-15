"""
PropertyMatching experiments for model: gpt-oss-20b
Auto-generated script - runs 2c and 2d experiments with shuffled properties.
"""

import os
import sys
import ast
import json
import time
import random
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import dspy
from easy_llm_importer import create_client, DSPyAdapter

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "gpt-oss-20b"
OUTPUT_DIR = os.path.join(script_dir, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"Starting PropertyMatching experiments for: {MODEL_NAME}")
print(f"{'='*60}")

# ============================================================================
# CREATE CLIENT AND CONFIGURE DSPY
# ============================================================================
client = create_client()
adapter = DSPyAdapter(client, model_name=MODEL_NAME)
lm = adapter.get_dspy_lm()
dspy.settings.configure(lm=lm)
print(f"Configured DSPy with model: {MODEL_NAME}")

# ============================================================================
# LOAD DATA
# ============================================================================
data_path = os.path.join(parent_dir, '..', '..', 'data', 'SCAR_cleaned_manually.csv')
df_scar = pd.read_csv(data_path)

# Parse mappings
df_scar['mappings_list'] = df_scar['mappings_parsed'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x else []
)

# Extract properties
df_scar['properties_unfamiliar'] = df_scar['mappings_list'].apply(
    lambda x: [pair[0] for pair in x] if x else []
)
df_scar['properties_familiar'] = df_scar['mappings_list'].apply(
    lambda x: [pair[1] for pair in x] if x else []
)

# Clean concept names
df_scar['unfamiliar_concept'] = df_scar['system_a'].str.strip()
df_scar['familiar_concept'] = df_scar['system_b'].str.strip()

# Use background descriptions
df_scar['description_unfamiliar'] = df_scar['system_a_background']
df_scar['description_familiar'] = df_scar['system_b_background']

print(f"Loaded {len(df_scar)} rows from SCAR dataset")

# ============================================================================
# DSPY SIGNATURES
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


def run_property_matching_no_desc(row):
    """Run PropertyMatching (no description) on a single row with SHUFFLED properties"""
    predictor = dspy.ChainOfThought(PropertyMatching)
    
    # Shuffle unfamiliar property list (deterministic per row for reproducibility)
    random.seed(row['id'])
    unfamiliar_props_shuffled = row['properties_unfamiliar'].copy()
    familiar_props_shuffled = row['properties_familiar'].copy()
    random.shuffle(unfamiliar_props_shuffled)
    
    result = predictor(
        unfamiliar_concept=row['unfamiliar_concept'],
        properties_of_unfamiliar_concept=unfamiliar_props_shuffled,
        familiar_concept=row['familiar_concept'],
        properties_of_familiar_concept=familiar_props_shuffled
    )
    reasoning = result.reasoning
    return result.mapped_source_properties, reasoning


def run_property_matching_with_desc(row):
    """Run PropertyMatchingWithDescription on a single row with SHUFFLED properties"""
    predictor = dspy.ChainOfThought(PropertyMatchingWithDescription)
    
    # Shuffle unfamiliar property list (deterministic per row for reproducibility)
    random.seed(row['id'])
    unfamiliar_props_shuffled = row['properties_unfamiliar'].copy()
    familiar_props_shuffled = row['properties_familiar'].copy()
    random.shuffle(unfamiliar_props_shuffled)
    
    result = predictor(
        unfamiliar_concept=row['unfamiliar_concept'],
        description_of_unfamiliar_concept=row['description_unfamiliar'],
        properties_of_unfamiliar_concept=unfamiliar_props_shuffled,
        familiar_concept=row['familiar_concept'],
        description_of_familiar_concept=row['description_familiar'],
        properties_of_familiar_concept=familiar_props_shuffled
    )
    reasoning = result.reasoning    
    return result.mapped_source_properties, reasoning


def run_experiment(df, experiment_name, experiment_func, output_file):
    """Run an experiment for all rows and save results"""
    print(f"\n--- Running {experiment_name} ---")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=experiment_name):
        start_time = time.time()
        
        try:
            result, reasoning = experiment_func(row)
            duration = time.time() - start_time
            
            ground_truth_mappings = dict(zip(row['properties_unfamiliar'], row['properties_familiar']))
            accuracy_metrics = calculate_mapping_accuracy(ground_truth_mappings, result)
            
            result_row = {
                'id': row['id'],
                'unfamiliar_concept': row['unfamiliar_concept'],
                'familiar_concept': row['familiar_concept'],
                'model': MODEL_NAME,
                'success': True,
                'error': None,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'ground_truth_properties_unfamiliar': row['properties_unfamiliar'],
                'ground_truth_properties_familiar': row['properties_familiar'],
                'ground_truth_mappings': ground_truth_mappings,
                'predicted_mappings': result,
                'reasoning': reasoning,
                'system_accuracy': accuracy_metrics['system_accuracy'],
                'correct_mappings': accuracy_metrics['correct_mappings'],
                'total_mappings': accuracy_metrics['total_mappings'],
                'concept_mapping_accuracy': accuracy_metrics['concept_mapping_accuracy']
            }
        except Exception as e:
            duration = time.time() - start_time
            ground_truth_mappings = dict(zip(row['properties_unfamiliar'], row['properties_familiar']))
            
            result_row = {
                'id': row['id'],
                'unfamiliar_concept': row['unfamiliar_concept'],
                'familiar_concept': row['familiar_concept'],
                'model': MODEL_NAME,
                'success': False,
                'error': str(e),
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'ground_truth_properties_unfamiliar': row['properties_unfamiliar'],
                'ground_truth_properties_familiar': row['properties_familiar'],
                'ground_truth_mappings': ground_truth_mappings,
                'predicted_mappings': None,
                'reasoning': None,
                'system_accuracy': False,
                'correct_mappings': 0,
                'total_mappings': len(ground_truth_mappings),
                'concept_mapping_accuracy': 0.0
            }
        
        results.append(result_row)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Calculate summary
    success_count = sum(1 for r in results if r['success'])
    system_accuracy = sum(1 for r in results if r['system_accuracy']) / len(results) * 100
    avg_concept_accuracy = sum(r['concept_mapping_accuracy'] for r in results) / len(results) * 100
    
    print(f"{experiment_name} complete: {success_count}/{len(results)} success, "
          f"System Acc: {system_accuracy:.1f}%, Concept Acc: {avg_concept_accuracy:.1f}%")
    print(f"Saved to: {output_file}")
    
    return results


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

if __name__ == "__main__":
    start_total = time.time()
    
    # Experiment 2c: PropertyMatching (No Description)
    output_2c = os.path.join(OUTPUT_DIR, f"2c_{MODEL_NAME.replace('/', '_')}.json")
    run_experiment(df_scar, "2c PropertyMatching (No Desc)", run_property_matching_no_desc, output_2c)
    
    # Experiment 2d: PropertyMatching (With Description)
    output_2d = os.path.join(OUTPUT_DIR, f"2d_{MODEL_NAME.replace('/', '_')}.json")
    run_experiment(df_scar, "2d PropertyMatching (With Desc)", run_property_matching_with_desc, output_2d)
    
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"COMPLETED: {MODEL_NAME}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")
