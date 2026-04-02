"""
Rerun Failed WithSub Models
Checks existing withsub CSV files and reruns only failed models (withsub mode + evaluation)
"""

import os
import sys
import json
import pandas as pd
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

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
from core.config import MODEL_LIST, RESULTS_DIR, get_output_path


def is_empty_analogies(analogies_json: str) -> bool:
    """Check if generated_analogies is empty."""
    if pd.isna(analogies_json) or not analogies_json:
        return True
    try:
        analogies = json.loads(analogies_json)
        return len(analogies) == 0 or all(not a or a.strip() == "" for a in analogies)
    except:
        return True


def is_empty_subconcepts(subconcepts_json: str) -> bool:
    """Check if analogy_subconcepts is empty or all empty strings."""
    if pd.isna(subconcepts_json) or not subconcepts_json:
        return True
    try:
        subconcepts = json.loads(subconcepts_json)
        if len(subconcepts) == 0:
            return True
        # Check if all or most (>80%) are empty strings
        empty_count = sum(1 for sc in subconcepts if not sc or str(sc).strip() == "")
        return empty_count / len(subconcepts) > 0.8 if len(subconcepts) > 0 else True
    except:
        return True


def count_empty_subconcepts(subconcepts_json: str) -> Tuple[int, int]:
    """
    Count empty and total subconcepts in a record.
    
    Returns:
        (empty_count, total_count)
    """
    if pd.isna(subconcepts_json) or not subconcepts_json:
        return 0, 0
    try:
        subconcepts = json.loads(subconcepts_json)
        if len(subconcepts) == 0:
            return 0, 0
        empty_count = sum(1 for sc in subconcepts if not sc or str(sc).strip() == "")
        return empty_count, len(subconcepts)
    except:
        return 0, 0


def check_model_failed(model_name: str) -> Tuple[bool, str]:
    """
    Check if a model's withsub results are failed/empty.
    
    Returns:
        (is_failed, reason)
    """
    file_path = get_output_path(model_name, "withsub", is_eval=False)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return True, "File does not exist"
    
    try:
        df = pd.read_csv(file_path)
        
        if len(df) == 0:
            return True, "File is empty"
        
        # Check error rate
        if 'status' in df.columns:
            error_count = (df['status'] != 'success').sum()
            error_rate = error_count / len(df)
            if error_rate > 0.5:
                return True, f"High error rate: {error_rate:.1%}"
        
        # Check for empty analogies
        if 'generated_analogies' in df.columns:
            empty_analogies = df['generated_analogies'].apply(is_empty_analogies).sum()
            if empty_analogies > len(df) * 0.5:
                return True, f"Empty analogies: {empty_analogies}/{len(df)} records"
        
        # Check for empty subconcepts
        if 'analogy_subconcepts' in df.columns:
            # Check 1: Records with >80% empty subconcepts
            records_with_empty = df['analogy_subconcepts'].apply(is_empty_subconcepts).sum()
            if records_with_empty > len(df) * 0.5:
                return True, f"Empty subconcepts: {records_with_empty}/{len(df)} records"
            
            # Check 2: Overall percentage of empty subconcepts across all records
            total_empty = 0
            total_subconcepts = 0
            for _, row in df.iterrows():
                empty_count, total_count = count_empty_subconcepts(row.get('analogy_subconcepts', ''))
                total_empty += empty_count
                total_subconcepts += total_count
            
            if total_subconcepts > 0:
                empty_percentage = total_empty / total_subconcepts
                # If >70% of all subconcepts are empty, mark as failed
                if empty_percentage > 0.7:
                    return True, f"High empty subconcept rate: {empty_percentage:.1%} ({total_empty}/{total_subconcepts} empty)"
        
        return False, "OK"
        
    except Exception as e:
        return True, f"Error reading file: {str(e)}"


def rerun_model_withsub(model_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Rerun withsub mode for a model (generation + evaluation).
    
    Returns:
        (success, message)
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # LLM folder
    core_dir = os.path.join(script_dir, "core")
    
    # Step 1: Run generation
    if verbose:
        print(f"  [1/2] Generating {model_name} - withsub...")
    
    gen_result = subprocess.run(
        [sys.executable, "run_model.py", "--model", model_name, "--mode", "withsub"],
        cwd=core_dir,
        capture_output=True,
        text=True
    )
    
    if gen_result.returncode != 0:
        return False, f"Generation failed: {gen_result.stderr[:200]}"
    
    # Step 2: Run evaluation
    if verbose:
        print(f"  [2/2] Evaluating {model_name} - withsub...")
    
    eval_file = get_output_path(model_name, "withsub", is_eval=False)
    if not os.path.exists(eval_file):
        return False, "Generation file not found after generation"
    
    eval_result = subprocess.run(
        [sys.executable, "evaluate_model.py", "--input", eval_file],
        cwd=core_dir,
        capture_output=True,
        text=True
    )
    
    if eval_result.returncode != 0:
        return False, f"Evaluation failed: {eval_result.stderr[:200]}"
    
    return True, "Success"


def main():
    """Main entry point."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Check for failed withsub models")
    parser.add_argument("--check-only", action="store_true", help="Only check and list failed models (no rerun)")
    parser.add_argument("--list-only", action="store_true", help="Only list failed models (no confirmation)")
    args = parser.parse_args()
    
    # Check all models
    failed_models = []
    ok_models = []
    
    if not args.list_only:
        print("=" * 70)
        print("Checking for Failed WithSub Models")
        print("=" * 70)
        print()
        print("Checking models...")
    
    for model in MODEL_LIST:
        is_failed, reason = check_model_failed(model)
        if is_failed:
            failed_models.append((model, reason))
            if not args.list_only:
                print(f"  [FAILED] {model}: {reason}")
        else:
            ok_models.append(model)
            if not args.list_only:
                print(f"  [OK] {model}")
    
    if not args.list_only:
        print()
        print("=" * 70)
        print(f"Summary: {len(ok_models)} OK, {len(failed_models)} failed")
        print("=" * 70)
    
    if len(failed_models) == 0:
        if not args.list_only:
            print("\nNo failed models found. All withsub results are OK!")
        return
    
    if args.check_only or args.list_only:
        # Just list failed models and exit
        # For --list-only, output only model names (one per line) for easy parsing
        if args.list_only:
            # Output to stderr for status, stdout for model names only
            sys.stderr.write(f"Found {len(failed_models)} failed model(s)\n")
            for model, reason in failed_models:
                print(model)  # stdout - just model names
        else:
            print(f"\nFailed models ({len(failed_models)}):")
            for model, reason in failed_models:
                print(f"  - {model}: {reason}")
        return
    
    print(f"\nFailed models to rerun ({len(failed_models)}):")
    for model, reason in failed_models:
        print(f"  - {model}: {reason}")
    
    # Ask for confirmation
    print()
    response = input("Rerun these models? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Rerun failed models (sequential - for reference, but PowerShell script uses parallel)
    print()
    print("=" * 70)
    print("Rerunning Failed Models (Sequential)")
    print("NOTE: Use rerun_withsub_failed.ps1 for parallel execution")
    print("=" * 70)
    print()
    
    results = []
    for model, reason in failed_models:
        print(f"\nRerunning: {model} (reason: {reason})")
        success, message = rerun_model_withsub(model, verbose=True)
        results.append((model, success, message))
        if success:
            print(f"  ✓ {model} completed successfully")
        else:
            print(f"  ✗ {model} failed: {message}")
    
    # Final summary
    print()
    print("=" * 70)
    print("Rerun Complete")
    print("=" * 70)
    print()
    
    successful = [m for m, s, _ in results if s]
    failed = [m for m, s, _ in results if not s]
    
    print(f"Successfully rerun: {len(successful)}/{len(results)}")
    if successful:
        print("  Models:", ", ".join(successful))
    
    if failed:
        print(f"\nFailed to rerun: {len(failed)}/{len(results)}")
        print("  Models:", ", ".join(failed))
        for model, _, message in results:
            if not any(m == model for m, s, _ in results if s):
                print(f"    {model}: {message}")


if __name__ == "__main__":
    main()
