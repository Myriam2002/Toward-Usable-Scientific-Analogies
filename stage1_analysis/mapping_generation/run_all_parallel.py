"""
Parallel Launcher: Execute all 12 model scripts simultaneously.
Each script runs as a separate Python process with its own DSPy settings.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARALLEL_RUNNERS_DIR = os.path.join(SCRIPT_DIR, 'parallel_runners')

# All model script files
MODEL_SCRIPTS = [
    'run_gpt-oss-20b.py',
    'run_gpt-oss-120b.py',
    'run_gpt-4_1-mini.py',
    'run_gpt-4_1-nano.py',
    'run_grok-4-fast.py',
    'run_gemini-2_5-flash-lite.py',
    'run_llama-3_1-405b-instruct.py',
    'run_meta-llama-3-1-70b-instruct.py',
    'run_meta-llama-3-1-8b-instruct.py',
    'run_deepseek-r1.py',
    'run_qwen3-14b.py',
    'run_qwen3-32b.py'
]


def run_all_parallel():
    """Launch all model scripts in parallel and wait for completion"""
    print("=" * 70)
    print("PARALLEL MODEL EXECUTION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to run: {len(MODEL_SCRIPTS)}")
    print(f"Scripts directory: {PARALLEL_RUNNERS_DIR}")
    print("=" * 70)
    
    # Verify all scripts exist
    missing = []
    for script in MODEL_SCRIPTS:
        script_path = os.path.join(PARALLEL_RUNNERS_DIR, script)
        if not os.path.exists(script_path):
            missing.append(script)
    
    if missing:
        print(f"\nERROR: Missing scripts: {missing}")
        print("Run 'python generate_scripts.py' first to create them.")
        return False
    
    print("\nLaunching all scripts in parallel...")
    
    # Start all processes
    processes = {}
    start_time = time.time()
    
    for script in MODEL_SCRIPTS:
        script_path = os.path.join(PARALLEL_RUNNERS_DIR, script)
        model_name = script.replace('run_', '').replace('.py', '')
        
        # Create log file for this model
        log_file = os.path.join(PARALLEL_RUNNERS_DIR, 'outputs', f'{model_name}_log.txt')
        
        # Launch the script
        with open(log_file, 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=PARALLEL_RUNNERS_DIR
            )
            processes[model_name] = {
                'process': proc,
                'script': script,
                'log_file': log_file,
                'start_time': time.time()
            }
        
        print(f"  Started: {model_name} (PID: {proc.pid})")
    
    print(f"\nAll {len(processes)} processes started. Waiting for completion...")
    print("-" * 70)
    
    # Monitor and wait for all processes
    completed = set()
    while len(completed) < len(processes):
        for model_name, info in processes.items():
            if model_name in completed:
                continue
            
            proc = info['process']
            return_code = proc.poll()
            
            if return_code is not None:
                # Process finished
                elapsed = time.time() - info['start_time']
                status = "SUCCESS" if return_code == 0 else f"FAILED (code {return_code})"
                print(f"  {status}: {model_name} ({elapsed:.1f}s)")
                completed.add(model_name)
        
        if len(completed) < len(processes):
            time.sleep(2)  # Check every 2 seconds
    
    # Summary
    total_time = time.time() - start_time
    
    print("-" * 70)
    print("\nCOMPLETION SUMMARY")
    print("=" * 70)
    
    success_count = 0
    fail_count = 0
    
    for model_name, info in processes.items():
        return_code = info['process'].returncode
        elapsed = time.time() - info['start_time']
        
        if return_code == 0:
            success_count += 1
            status_icon = "✓"
        else:
            fail_count += 1
            status_icon = "✗"
        
        print(f"  {status_icon} {model_name}: exit code {return_code} ({elapsed:.1f}s)")
    
    print("-" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful: {success_count}/{len(processes)}")
    print(f"Failed: {fail_count}/{len(processes)}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if fail_count > 0:
        print("\nCheck log files for failed models:")
        for model_name, info in processes.items():
            if info['process'].returncode != 0:
                print(f"  {info['log_file']}")
    
    print("\nOutput files saved in:", os.path.join(PARALLEL_RUNNERS_DIR, 'outputs'))
    print("Run 'python aggregate_results.py' to combine results into final CSVs.")
    
    return fail_count == 0


if __name__ == "__main__":
    success = run_all_parallel()
    sys.exit(0 if success else 1)


