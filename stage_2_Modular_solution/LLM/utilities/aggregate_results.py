"""
Aggregate Results Script - Combine all model evaluation results by mode
Creates two aggregated files: all_results_targetonly.csv and all_results_withsub.csv
"""

import os
import sys
import json
import pandas as pd
import glob
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
env_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..', '.env'),
    os.path.join(os.path.dirname(__file__), '..', '.env'),
    os.path.join(os.path.dirname(__file__), '.env'),
]
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

# Add core to path and import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.config import RESULTS_DIR, MODEL_LIST


def aggregate_results():
    """
    Aggregate all evaluation results by mode.
    Creates:
    - results/all_results_targetonly.csv
    - results/all_results_withsub.csv
    """
    print("=" * 70)
    print("AGGREGATING ALL RESULTS")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Expected models: {len(MODEL_LIST)}")
    print()
    
    # Find all evaluation files
    eval_files = glob.glob(os.path.join(RESULTS_DIR, "LLM_*_eval.csv"))
    print(f"Found {len(eval_files)} evaluation files")
    
    if len(eval_files) == 0:
        print("\nERROR: No evaluation files found!")
        print("Make sure you've run the generation and evaluation pipeline first.")
        return None, None
    
    # Separate by mode
    targetonly_files = []
    withsub_files = []
    
    for f in eval_files:
        filename = os.path.basename(f)
        if '_targetonly_eval.csv' in filename:
            targetonly_files.append(f)
        elif '_withsub_eval.csv' in filename:
            withsub_files.append(f)
    
    print(f"\nFiles by mode:")
    print(f"  targetonly: {len(targetonly_files)}")
    print(f"  withsub:    {len(withsub_files)}")
    
    # Process each mode
    results = {}
    
    for mode, files in [('targetonly', targetonly_files), ('withsub', withsub_files)]:
        print(f"\n{'-'*70}")
        print(f"Processing {mode} mode ({len(files)} files)")
        print("-" * 70)
        
        if len(files) == 0:
            print(f"  No {mode} files found, skipping...")
            results[mode] = None
            continue
        
        all_dfs = []
        
        for filepath in sorted(files):
            filename = os.path.basename(filepath)
            
            # Extract model name from filename: LLM_{model}_{mode}_eval.csv
            # Remove prefix and suffix
            model_part = filename.replace('LLM_', '').replace(f'_{mode}_eval.csv', '')
            model_name = model_part
            
            try:
                df = pd.read_csv(filepath)
                df['model'] = model_name
                df['mode'] = mode
                all_dfs.append(df)
                print(f"  [OK] {model_name}: {len(df)} records")
            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}")
        
        if len(all_dfs) == 0:
            print(f"  No valid data for {mode}")
            results[mode] = None
            continue
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save aggregated file
        output_path = os.path.join(RESULTS_DIR, f'all_results_{mode}.csv')
        combined_df.to_csv(output_path, index=False)
        
        results[mode] = combined_df
        
        print(f"\n  Combined: {len(combined_df)} total records")
        print(f"  Models:   {combined_df['model'].nunique()}")
        print(f"  Saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70)
    
    for mode in ['targetonly', 'withsub']:
        df = results.get(mode)
        if df is not None:
            output_path = os.path.join(RESULTS_DIR, f'all_results_{mode}.csv')
            print(f"\n{mode}:")
            print(f"  File: {output_path}")
            print(f"  Records: {len(df)}")
            print(f"  Models: {df['model'].nunique()}")
            
            # Quick stats if available
            if 'judge_baseline' in df.columns:
                successful = df[df['status'] == 'success']
                if len(successful) > 0:
                    # Extract averages from judge dictionaries
                    def get_judge_avg(judge_json):
                        try:
                            judge = json.loads(judge_json)
                            return judge.get('average', 0)
                        except:
                            return 0
                    
                    baseline_avgs = successful['judge_baseline'].apply(get_judge_avg)
                    embedding_avgs = successful['judge_embedding'].apply(get_judge_avg)
                    
                    print(f"  Avg Judge Score (Baseline):  {baseline_avgs.mean():.2f}")
                    print(f"  Avg Judge Score (Embedding): {embedding_avgs.mean():.2f}")
    
    print("\n" + "=" * 70)
    
    return results.get('targetonly'), results.get('withsub')


def main():
    """Main entry point."""
    aggregate_results()


if __name__ == "__main__":
    main()
