"""
Aggregate Results: Combine 24 JSON files into 2 final CSV files.
- 12 x 2c_*.json -> 2c_property_matching_no_desc.csv
- 12 x 2d_*.json -> 2d_property_matching_with_desc.csv
"""

import os
import json
import pandas as pd
from datetime import datetime

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'parallel_runners', 'outputs')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def aggregate_experiment(experiment_prefix, output_csv_name):
    """
    Aggregate all JSON files for an experiment into a single CSV.
    
    Args:
        experiment_prefix: '2c' or '2d'
        output_csv_name: Name of the output CSV file
    
    Returns:
        DataFrame of combined results, or None if no files found
    """
    print(f"\n--- Aggregating {experiment_prefix} files ---")
    
    all_results = []
    files_processed = 0
    
    # Find all matching JSON files
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if filename.startswith(f"{experiment_prefix}_") and filename.endswith('.json'):
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # Convert complex types to JSON strings for CSV storage
                for r in results:
                    r['ground_truth_properties_unfamiliar'] = json.dumps(r['ground_truth_properties_unfamiliar'])
                    r['ground_truth_properties_familiar'] = json.dumps(r['ground_truth_properties_familiar'])
                    r['ground_truth_mappings'] = json.dumps(r['ground_truth_mappings'])
                    r['predicted_mappings'] = json.dumps(r['predicted_mappings']) if r['predicted_mappings'] else None
                
                all_results.extend(results)
                files_processed += 1
                
                # Extract model name from filename
                model_name = filename.replace(f"{experiment_prefix}_", "").replace(".json", "")
                row_count = len(results)
                success_count = sum(1 for r in results if r['success'])
                
                print(f"  {filename}: {row_count} rows ({success_count} success)")
                
            except Exception as e:
                print(f"  ERROR reading {filename}: {e}")
    
    if not all_results:
        print(f"  No files found for {experiment_prefix}_*.json")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(RESULTS_DIR, output_csv_name)
    df.to_csv(output_path, index=False)
    
    print(f"\n  Combined {files_processed} files -> {len(all_results)} total rows")
    print(f"  Saved to: {output_path}")
    
    return df


def print_summary(df, experiment_name):
    """Print summary statistics for a DataFrame"""
    if df is None:
        return
    
    print(f"\n  Summary for {experiment_name}:")
    print(f"    Total rows: {len(df)}")
    print(f"    Unique models: {df['model'].nunique()}")
    
    # Success rate
    success_rate = df['success'].mean() * 100
    print(f"    Success rate: {success_rate:.1f}%")
    
    # System accuracy (% of rows where all mappings correct)
    system_acc = df['system_accuracy'].mean() * 100
    print(f"    System accuracy: {system_acc:.1f}%")
    
    # Average concept mapping accuracy
    avg_concept_acc = df['concept_mapping_accuracy'].mean() * 100
    print(f"    Avg concept mapping accuracy: {avg_concept_acc:.1f}%")
    
    # Per-model breakdown
    print(f"\n    Per-model results:")
    model_summary = df.groupby('model').agg({
        'success': 'mean',
        'system_accuracy': 'mean',
        'concept_mapping_accuracy': 'mean'
    }).round(3)
    
    model_summary.columns = ['Success Rate', 'System Acc', 'Concept Acc']
    model_summary = model_summary.sort_values('Concept Acc', ascending=False)
    
    for model, row in model_summary.iterrows():
        print(f"      {model}: "
              f"Success={row['Success Rate']*100:.1f}%, "
              f"SysAcc={row['System Acc']*100:.1f}%, "
              f"ConceptAcc={row['Concept Acc']*100:.1f}%")


def aggregate_all():
    """Aggregate results for both experiments"""
    print("=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {OUTPUT_DIR}")
    print(f"Output directory: {RESULTS_DIR}")
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"\nERROR: Output directory does not exist: {OUTPUT_DIR}")
        print("Run 'python run_all_parallel.py' first to generate results.")
        return False
    
    # List available files
    json_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    print(f"\nFound {len(json_files)} JSON files")
    
    # Aggregate 2c
    df_2c = aggregate_experiment('2c', '2c_property_matching_no_desc.csv')
    print_summary(df_2c, '2c PropertyMatching (No Description)')
    
    # Aggregate 2d
    df_2d = aggregate_experiment('2d', '2d_property_matching_with_desc.csv')
    print_summary(df_2d, '2d PropertyMatching (With Description)')
    
    print("\n" + "=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70)
    
    if df_2c is not None:
        print(f"  2c: {len(df_2c)} rows -> results/2c_property_matching_no_desc.csv")
    if df_2d is not None:
        print(f"  2d: {len(df_2d)} rows -> results/2d_property_matching_with_desc.csv")
    
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    aggregate_all()


