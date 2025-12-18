"""
Script to run RAG source finder evaluation for all embedding modes
and generate visualizations
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rag_source_finder import RAGSourceFinder
from rag_visualization import ComprehensiveAnalyzer

# Load environment variables
load_dotenv('../../.env')

# Configuration
DATA_PATH = '../../data/SCAR_cleaned_manually.csv'
RESULTS_DIR = "results"
TOP_K = 20

# All embedding modes to test
EMBEDDING_MODES = ["name_only", "name_background", "name_properties", "name_properties_background"]

def main():
    print("=" * 70)
    print("RAG Source Finder - Full Evaluation")
    print("=" * 70)
    print(f"Data path: {DATA_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Top-K: {TOP_K}")
    print(f"Embedding modes: {EMBEDDING_MODES}")
    print("=" * 70)
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run evaluation for each mode
    print("\n" + "=" * 70)
    print("STEP 1: Running RAG Evaluation for All Modes")
    print("=" * 70)
    
    for mode in EMBEDDING_MODES:
        print(f"\n{'='*20} Testing Mode: {mode} {'='*20}")
        
        try:
            # Initialize RAG finder with specific mode
            print(f"Initializing RAG Source Finder with mode: {mode}")
            rag_finder = RAGSourceFinder(embedding_mode=mode)
            
            # Load corpus
            print("Loading corpus...")
            rag_finder.load_corpus_from_csv(DATA_PATH)
            
            # Embed corpus
            print("Embedding corpus (this may take a minute)...")
            rag_finder.embed_corpus()
            
            print("RAG finder ready!")
            
            # Run RAG evaluation
            print("Running RAG evaluation...")
            rag_results = rag_finder.evaluate_on_dataset(
                DATA_PATH,
                top_k=TOP_K
            )
            
            print(f"Completed! Evaluated {len(rag_results)} examples")
            
            # Save results (with retry logic for file locks)
            mode_filename = f"rag_results_{mode}.csv"
            rag_results_path = os.path.join(RESULTS_DIR, mode_filename)
            
            # Try to save, with retry if file is locked
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    rag_results.to_csv(rag_results_path, index=False)
                    print(f"✅ Saved {mode} results to {rag_results_path}")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️  File is locked (attempt {attempt + 1}/{max_retries}). Please close the file and press Enter to retry...")
                        input()  # Wait for user to close the file
                    else:
                        print(f"❌ Could not save {mode} results: {e}")
                        print(f"   Please close {rag_results_path} and run the script again, or save manually.")
                        # Save with a different name as backup
                        backup_path = rag_results_path.replace('.csv', '_new.csv')
                        try:
                            rag_results.to_csv(backup_path, index=False)
                            print(f"   Saved backup to {backup_path}")
                        except:
                            pass
            
            # Display sample results
            print(f"\nSample Results for Mode '{mode}':")
            print("-" * 50)
            for idx, row in rag_results.head(3).iterrows():
                print(f"Target: {row['target']}")
                print(f"Gold Source: {row['gold_source']}")
                print(f"Predicted (Rank 1): {row['predicted_rank_1']}")
                print(f"Gold Rank: {row['gold_rank']}")
                if row['target_properties']:
                    print(f"Target Properties: {row['target_properties'][:100]}...")
                print(f"Top 3 sources: {row['top_k_sources'][:3]}")
                print()
            
        except Exception as e:
            print(f"❌ Error processing mode {mode}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Run visualization
    print("\n" + "=" * 70)
    print("STEP 2: Generating Visualizations")
    print("=" * 70)
    
    try:
        analyzer = ComprehensiveAnalyzer(
            results_dir=RESULTS_DIR,
            output_dir=os.path.join(RESULTS_DIR, "visualizations")
        )
        analyzer.run_full_analysis()
        print("\n✅ Visualization complete!")
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ All done! Check the results directory for CSV files and visualizations.")
    print("=" * 70)

if __name__ == "__main__":
    main()

