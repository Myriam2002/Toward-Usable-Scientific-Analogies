"""
Rerun Problematic Records
Identifies records with problems across all models and reruns only those records.
"""

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Set, Tuple
from pathlib import Path

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
from core.config import MODEL_LIST, SCAR_PATH, RESULTS_DIR, get_output_path


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
    """Check if analogy_subconcepts is empty or mostly empty strings."""
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
    """Count empty and total subconcepts in a record."""
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


def is_record_problematic(record: pd.Series, mode: str) -> Tuple[bool, str]:
    """
    Check if a single record has problems.
    
    Returns:
        (is_problematic, reason)
    """
    # Check status
    if 'status' in record:
        if record['status'] != 'success':
            return True, f"Status: {record['status']}"
    
    # Check for empty analogies
    if 'generated_analogies' in record:
        if is_empty_analogies(record['generated_analogies']):
            return True, "Empty analogies"
    
    # Check for empty subconcepts (only in withsub mode)
    if mode == "withsub" and 'analogy_subconcepts' in record:
        if is_empty_subconcepts(record['analogy_subconcepts']):
            return True, "Empty subconcepts"
        
        # Also check overall empty rate
        empty_count, total_count = count_empty_subconcepts(record['analogy_subconcepts'])
        if total_count > 0:
            empty_rate = empty_count / total_count
            if empty_rate > 0.7:
                return True, f"High empty subconcept rate: {empty_rate:.1%}"
    
    return False, "OK"


def find_problematic_records(model: str, mode: str) -> List[Tuple[str, str]]:
    """
    Find all problematic records for a model/mode combination.
    
    Returns:
        List of (target, reason) tuples for problematic records
    """
    file_path = get_output_path(model, mode, is_eval=False)
    
    if not os.path.exists(file_path):
        return []
    
    try:
        df = pd.read_csv(file_path)
        problematic = []
        
        for idx, row in df.iterrows():
            is_problem, reason = is_record_problematic(row, mode)
            if is_problem:
                target = row.get('target', f"Record_{idx}")
                problematic.append((target, reason))
        
        return problematic
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def get_all_problematic_records() -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Find problematic records across all models and modes.
    
    Returns:
        Dict[model][mode] = List[(target, reason)]
    """
    all_problems = {}
    
    print("Scanning all models for problematic records...")
    for model in MODEL_LIST:
        all_problems[model] = {}
        for mode in ["targetonly", "withsub"]:
            problems = find_problematic_records(model, mode)
            all_problems[model][mode] = problems
            if problems:
                print(f"  {model} ({mode}): {len(problems)} problematic records")
    
    return all_problems


def get_unique_problematic_targets(all_problems: Dict) -> Set[str]:
    """Extract unique target concepts that have problems in any model/mode."""
    unique_targets = set()
    for model, modes in all_problems.items():
        for mode, problems in modes.items():
            for target, reason in problems:
                unique_targets.add(target)
    return unique_targets


def rerun_records_for_model(model: str, mode: str, target_list: List[str], verbose: bool = False):
    """
    Rerun specific records (by target) for a model/mode.
    Uses run_model.py with --targets parameter to filter records.
    """
    import subprocess
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(target_list) == 0:
        return True, "No targets to rerun"
    
    try:
        # Run generation with target filter (run_model.py will merge results)
        cmd = [
            sys.executable, "run_model.py",
            "--model", model,
            "--mode", mode,
            "--targets", ",".join(target_list)
        ]
        
        if verbose:
            cmd.append("--verbose")
        
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False, f"Generation failed: {result.stderr[:200]}"
        
        return True, "Success"
        
    except Exception as e:
        return False, str(e)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rerun problematic records")
    parser.add_argument("--check-only", action="store_true", help="Only check and list problematic records")
    parser.add_argument("--export-targets", action="store_true", help="Export targets to files for parallel execution")
    parser.add_argument("--rerun", action="store_true", help="Rerun all problematic records (no confirmation)")
    parser.add_argument("--model", type=str, help="Rerun only for specific model")
    parser.add_argument("--mode", type=str, choices=["targetonly", "withsub"], help="Rerun only for specific mode")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Rerun Problematic Records")
    print("=" * 70)
    print()
    
    # Step 1: Find all problematic records
    all_problems = get_all_problematic_records()
    
    # Step 2: Summarize
    total_problems = sum(len(problems) for model in all_problems.values() 
                        for problems in model.values())
    
    if total_problems == 0:
        print("\nNo problematic records found!")
        return
    
    print()
    print("=" * 70)
    print(f"Summary: {total_problems} problematic records found")
    print("=" * 70)
    print()
    
    # Show breakdown (format for easy parsing by PowerShell)
    # If --export-targets, also save targets to files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for model in MODEL_LIST:
        for mode in ["targetonly", "withsub"]:
            problems = all_problems[model][mode]
            if problems:
                targets = [t for t, _ in problems]
                
                if args.export_targets:
                    # Export targets to a file for parallel execution
                    targets_file = os.path.join(script_dir, f"targets_{model}_{mode}.txt")
                    with open(targets_file, 'w', encoding='utf-8') as f:
                        for target in targets:
                            f.write(target + '\n')
                    # Format for PowerShell parsing: "model (mode): N records -> filename"
                    print(f"{model} ({mode}): {len(problems)} records -> targets_{model}_{mode}.txt")
                else:
                    # Normal output format
                    print(f"{model} ({mode}): {len(problems)} records")
                    for target, reason in problems[:5]:  # Show first 5
                        print(f"  - {target[:50]}: {reason}")
                    if len(problems) > 5:
                        print(f"  ... and {len(problems) - 5} more")
                    print()
    
    # Step 3: Get unique targets
    unique_targets = get_unique_problematic_targets(all_problems)
    if not args.export_targets:
        print(f"\nUnique problematic targets: {len(unique_targets)}")
    
    if args.check_only or args.export_targets:
        return
    
    # Step 4: Ask for confirmation (unless --rerun flag)
    if not args.rerun:
        print()
        response = input("Rerun all problematic records? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    
    # Step 5: Rerun problematic records
    print()
    print("=" * 70)
    print("Rerunning Problematic Records")
    print("=" * 70)
    print()
    
    # Filter to specific model/mode if requested
    models_to_process = [args.model] if args.model else MODEL_LIST
    modes_to_process = [args.mode] if args.mode else ["targetonly", "withsub"]
    
    results = []
    for model in models_to_process:
        for mode in modes_to_process:
            problems = all_problems[model][mode]
            if problems:
                targets = [t for t, _ in problems]
                print(f"\nRerunning {model} ({mode}): {len(targets)} records")
                success, message = rerun_records_for_model(model, mode, targets, verbose=True)
                results.append((model, mode, len(targets), success, message))
                if success:
                    print(f"  ✓ {model} ({mode}) completed")
                else:
                    print(f"  ✗ {model} ({mode}) failed: {message}")
            elif args.model and args.mode:
                # If specific model/mode requested but no problems, report it
                print(f"\n{model} ({mode}): No problematic records found")
    
    # Final summary
    print()
    print("=" * 70)
    print("Rerun Complete")
    print("=" * 70)
    print()
    
    successful = [(m, mo, n) for m, mo, n, s, _ in results if s]
    failed = [(m, mo, n, msg) for m, mo, n, s, msg in results if not s]
    
    print(f"Successfully rerun: {len(successful)}/{len(results)} model/mode combinations")
    if successful:
        for model, mode, count in successful:
            print(f"  ✓ {model} ({mode}): {count} records")
    
    if failed:
        print(f"\nFailed to rerun: {len(failed)}/{len(results)} model/mode combinations")
        for model, mode, count, message in failed:
            print(f"  ✗ {model} ({mode}): {count} records - {message}")
    
    print("\nNOTE: After rerunning, you should run evaluate_model.py to update evaluations.")


if __name__ == "__main__":
    main()
