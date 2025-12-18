"""
Quick script to re-run evaluation and save results with _fixed suffix
This avoids conflicts with open CSV files
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rag_source_finder import RAGSourceFinder

# Load environment variables
load_dotenv('../../.env')

# Configuration
DATA_PATH = '../../data/SCAR_cleaned_manually.csv'
RESULTS_DIR = "results"
TOP_K = 20

# All embedding modes to test
EMBEDDING_MODES = ["name_only", "name_background", "name_properties", "name_properties_background"]

def main():
    print("Saving results with _fixed suffix to avoid file conflicts...")
    print("=" * 70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for mode in EMBEDDING_MODES:
        print(f"\nProcessing mode: {mode}")
        
        try:
            # Initialize RAG finder
            rag_finder = RAGSourceFinder(embedding_mode=mode)
            
            # Load corpus
            print("  Loading corpus...")
            rag_finder.load_corpus_from_csv(DATA_PATH)
            
            # Embed corpus
            print("  Embedding corpus...")
            rag_finder.embed_corpus()
            
            # Run evaluation
            print("  Running evaluation...")
            rag_results = rag_finder.evaluate_on_dataset(DATA_PATH, top_k=TOP_K)
            
            # Save with _fixed suffix
            mode_filename = f"rag_results_{mode}_fixed.csv"
            rag_results_path = os.path.join(RESULTS_DIR, mode_filename)
            rag_results.to_csv(rag_results_path, index=False)
            print(f"  ✅ Saved to {rag_results_path}")
            
            # Show sample with properties
            print(f"\n  Sample result (showing properties are now included):")
            sample = rag_results.iloc[0]
            print(f"    Target: {sample['target']}")
            print(f"    Properties: {sample['target_properties'][:80]}..." if sample['target_properties'] else "    Properties: (empty)")
            print(f"    Query text: {sample['query_embedding_text'][:100]}...")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Done! New results saved with _fixed suffix.")
    print("You can now close the old CSV files and rename these if needed.")

if __name__ == "__main__":
    main()

