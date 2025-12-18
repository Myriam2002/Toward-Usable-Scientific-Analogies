"""
Combine Model Checkpoints

This script combines individual model checkpoint files into unified ALL_MODELS_combined files.
Useful when running models in parallel with --model flag.

Usage:
    python combine_checkpoints.py --setting unpaired_properties
    python combine_checkpoints.py --setting all
    python combine_checkpoints.py --setting none --output-dir custom/path
"""

import os
import sys
import json
import argparse
import pandas as pd
from glob import glob
from pathlib import Path


# Available experiment settings
EXPERIMENT_SETTINGS = [
    'none',
    'none_description',
    'unpaired_properties',
    'unpaired_properties_description',
    'paired_properties',
    'paired_properties_description'
]


def combine_setting_checkpoints(setting, output_dir):
    """
    Combine all model checkpoint files for a specific setting.
    
    Args:
        setting: Experiment setting name
        output_dir: Directory containing checkpoint files
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Combining checkpoints for setting: {setting}")
    print(f"{'='*80}")
    
    # Find all checkpoint CSV files for this setting
    checkpoint_pattern = os.path.join(output_dir, f'{setting}_*_checkpoint.csv')
    checkpoint_files = glob(checkpoint_pattern)
    
    # Filter out existing combined files
    checkpoint_files = [f for f in checkpoint_files if 'ALL_MODELS' not in f]
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found for setting '{setting}'")
        print(f"   Pattern searched: {checkpoint_pattern}")
        return False
    
    print(f"📁 Found {len(checkpoint_files)} model checkpoint files:")
    for f in sorted(checkpoint_files):
        print(f"   • {os.path.basename(f)}")
    
    try:
        # Load all checkpoint CSVs
        dfs = []
        for checkpoint_file in checkpoint_files:
            df = pd.read_csv(checkpoint_file)
            dfs.append(df)
            print(f"   ✓ Loaded {os.path.basename(checkpoint_file)}: {len(df)} rows")
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save combined CSV
        combined_csv_path = os.path.join(output_dir, f'{setting}_ALL_MODELS_combined.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n✅ Saved combined CSV: {combined_csv_path}")
        
        # Save combined JSON
        combined_json = combined_df.to_dict('records')
        combined_json_path = os.path.join(output_dir, f'{setting}_ALL_MODELS_combined.json')
        with open(combined_json_path, 'w') as f:
            json.dump(combined_json, f, indent=2)
        print(f"✅ Saved combined JSON: {combined_json_path}")
        
        # Print summary statistics
        print(f"\n📊 Combined Results Summary:")
        print(f"   • Total results: {len(combined_df)}")
        print(f"   • Number of models: {combined_df['model'].nunique()}")
        print(f"   • Models included: {', '.join(sorted(combined_df['model'].unique()))}")
        
        # Per-model summary
        print(f"\n📈 Per-Model Statistics:")
        summary = combined_df.groupby('model').agg({
            'row_index': 'count',
            'sbert_similarity': 'mean',
            'error': lambda x: sum(pd.notna(x) & (x != 'None') & (x != '') & (x != None))
        }).rename(columns={
            'row_index': 'total_rows',
            'sbert_similarity': 'mean_sbert',
            'error': 'errors'
        })
        summary['mean_sbert'] = summary['mean_sbert'].round(4)
        print(summary.to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Error combining checkpoints: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Combine individual model checkpoint files into ALL_MODELS_combined files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine checkpoints for specific setting
  python combine_checkpoints.py --setting unpaired_properties
  
  # Combine checkpoints for all settings
  python combine_checkpoints.py --setting all
  
  # Use custom output directory
  python combine_checkpoints.py --setting none --output-dir custom/path

Notes:
  - This script looks for files matching: {setting}_{model}_checkpoint.csv
  - Existing ALL_MODELS_combined files will be overwritten
  - Run this after all individual model runs have completed
        """
    )
    
    parser.add_argument(
        '--setting',
        type=str,
        choices=EXPERIMENT_SETTINGS + ['all'],
        required=True,
        help='Experiment setting to combine, or "all" to combine all settings'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints/explanation_generation',
        help='Directory containing checkpoint files (default: checkpoints/explanation_generation)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("CHECKPOINT COMBINER")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        print(f"❌ Error: Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    # Determine which settings to combine
    if args.setting == 'all':
        settings_to_combine = EXPERIMENT_SETTINGS
        print(f"Mode: Combining ALL settings ({len(settings_to_combine)})")
    else:
        settings_to_combine = [args.setting]
        print(f"Mode: Combining SINGLE setting ({args.setting})")
    
    # Combine checkpoints for each setting
    results = {}
    for setting in settings_to_combine:
        success = combine_setting_checkpoints(setting, args.output_dir)
        results[setting] = success
    
    # Final summary
    print("\n" + "="*80)
    print("COMBINATION SUMMARY")
    print("="*80)
    
    successful = [s for s, success in results.items() if success]
    failed = [s for s, success in results.items() if not success]
    
    if successful:
        print(f"✅ Successfully combined {len(successful)} setting(s):")
        for setting in successful:
            print(f"   • {setting}")
    
    if failed:
        print(f"\n❌ Failed to combine {len(failed)} setting(s):")
        for setting in failed:
            print(f"   • {setting}")
    
    print(f"\n🎯 All combined files saved to: {args.output_dir}")
    
    # Exit with appropriate code
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

