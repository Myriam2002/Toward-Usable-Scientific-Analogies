# Source Finding for Scientific Analogies

This module implements and compares multiple approaches for finding analogous source domains from a closed corpus, given only a target domain. It includes comprehensive evaluation, visualization, and analysis tools.

## Overview

### Problem
Given a target concept (e.g., "biological clock") and a closed corpus of potential source domains, find the most analogous source (e.g., "clock") that can be used to explain the target.

### Approaches Implemented

1. **RAG-Based (Embedding Similarity)**
   - Uses OpenAI embeddings to represent targets and sources
   - Retrieves sources via cosine similarity
   - Fast, no LLM reasoning, good baseline
   - Multiple embedding modes: name_only, name_background, name_properties, name_properties_background

2. **Tournament-Style Elimination**
   - LLM evaluates sources in batches (configurable batch size, default 50)
   - Winners advance to next round
   - Continues until single winner remains
   - Captures reasoning at each stage
   - **Optimized for speed**: Larger batches = fewer rounds

3. **Comprehensive Visualization & Analysis**
   - Publication-ready visualizations and metrics
   - Multiple golden answer analysis
   - Error pattern identification
   - Performance comparison across embedding modes

## Files

### Core Modules
- `rag_source_finder.py` - RAG/embedding-based approach
- `iterative_source_finder.py` - Tournament approach  
- `evaluate_source_finding.py` - Evaluation framework with metrics
- `rag_visualization.py` - Comprehensive visualization and analysis tool
- `source_analysis_v1.ipynb` - Main notebook for experiments

### Documentation
- `README.md` - This comprehensive guide
- `QUICK_REFERENCE.md` - Quick start and common tasks
- `VISUALIZATION_README.md` - Detailed visualization guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

### Data
- Uses `../../data/SCAR_cleaned_manually.csv` as corpus
- `system_a` = target concepts
- `system_b` = source domains (corpus)

## Installation

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn openai python-dotenv tqdm scipy

# Set up API keys in .env file or environment
export OPENAI_API_KEY="your-key"
export DEEPINFRA_API_KEY="your-key"  # For LLM-based approaches
```

## Quick Start

### 1. Run RAG Analysis
```bash
# Open the main notebook
jupyter notebook source_analysis_v1.ipynb

# Or run RAG analysis programmatically
python -c "
from rag_source_finder import RAGSourceFinder
finder = RAGSourceFinder()
finder.load_corpus_from_csv('../../data/SCAR_cleaned_manually.csv')
finder.embed_corpus()
results = finder.evaluate_on_dataset('../../data/SCAR_cleaned_manually.csv')
results.to_csv('results/rag_results.csv', index=False)
"
```

### 2. Generate Visualizations
```bash
# Run comprehensive analysis and visualizations
python rag_visualization.py

# This generates:
# - results/rag_comprehensive_metrics.csv
# - results/rag_analysis_summary.txt
# - results/visualizations/*.png (8 publication-ready plots)
```

### 3. View Results
- **Metrics**: Check `results/rag_comprehensive_metrics.csv`
- **Summary**: Read `results/rag_analysis_summary.txt`
- **Visualizations**: Browse `results/visualizations/` directory

## Configuration

Edit configuration in the notebook or script:

```python
# Test configuration
TEST_MODE = True  # Set False for full dataset
TEST_SAMPLE_SIZE = 10  # Number of examples
TOP_K = 20  # Top-K results to retrieve
TOURNAMENT_BATCH_SIZE = 50  # Sources per batch in tournament

# Models to test
TEST_MODELS = [
    "gpt-4.1-mini",
    "meta-llama-3-1-70b-instruct"
]

# Embedding modes to test
EMBEDDING_MODES = [
    "name_only",
    "name_background", 
    "name_properties",
    "name_properties_background"
]
```

## Usage Examples

### RAG Source Finding
```python
from rag_source_finder import RAGSourceFinder

# Initialize with specific embedding mode
finder = RAGSourceFinder(embedding_mode="name_properties")

# Load and embed corpus
finder.load_corpus_from_csv('../../data/SCAR_cleaned_manually.csv')
finder.embed_corpus()

# Find sources for a target
results = finder.find_source(
    target_name="biological clock",
    target_background="The biological clock regulates sleep-wake cycles.",
    target_properties="circadian rhythm, temporal regulation",
    top_k=10
)

# Evaluate on full dataset
evaluation_results = finder.evaluate_on_dataset(
    '../../data/SCAR_cleaned_manually.csv',
    top_k=20
)
```

### Tournament Source Finding
```python
from easy_llm_importer import LLMClient
from iterative_source_finder import TournamentSourceFinder

# Initialize LLM client
client = LLMClient()

# Create tournament finder
finder = TournamentSourceFinder(
    client=client,
    model="gpt-4.1-mini",
    batch_size=50
)

# Find best source
result = finder.find_source(
    target_name="biological clock",
    target_background="Description...",
    source_candidates=source_list
)
```

### Visualization and Analysis
```python
from rag_visualization import ComprehensiveAnalyzer

# Run complete analysis
analyzer = ComprehensiveAnalyzer(
    results_dir="results",
    output_dir="results/visualizations"
)
analyzer.run_full_analysis()

# Or use individual components
from rag_visualization import RAGDataLoader, RankingMetrics

loader = RAGDataLoader("results")
data = loader.load_all_results()

for mode, df in data.items():
    metrics = RankingMetrics.calculate_all_metrics(df)
    print(f"{mode}: MRR={metrics['MRR']:.4f}")
```

## Evaluation Metrics

### Ranking Metrics
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first golden answer
- **MAP (Mean Average Precision)**: Precision averaged across all golden answers
- **NDCG@20 (Normalized Discounted Cumulative Gain)**: Ranking quality with position discount
- **Hit@K**: Success rate within top-K results
- **Recall@K**: Proportion of all golden answers found in top-K

### Performance Metrics
- **Exact Match Accuracy**: Did the method find the correct source?
- **Top-K Accuracy**: Is the correct source in top-K? (K=1,3,5,10,20)
- **Average Rank**: Where does correct source typically appear?
- **Failure Rate**: Percentage of queries with no correct answer in top-K

### Multiple Golden Answer Analysis
- **Golden Coverage**: Proportion of all golden answers found
- **Avg Golden Per Target**: Average number of valid answers per query
- **Avg Found Per Target**: Average number successfully retrieved

## Generated Outputs

### Metrics Files
- `rag_comprehensive_metrics.csv` - All metrics in table format
- `rag_analysis_summary.txt` - Detailed text analysis and recommendations

### Visualizations (8 PNG files)
1. **Performance Heatmap** - All metrics across embedding modes
2. **Hit@K Curves** - Success rate at different thresholds
3. **Rank Distribution** - Histogram of where golden answers appear
4. **Metric Comparison** - MRR/MAP/NDCG bar charts
5. **Golden Coverage** - Multiple answer analysis
6. **Score Distribution** - Similarity scores for hits vs misses
7. **Failure Analysis** - Performance categories
8. **Recall Curves** - Coverage at different thresholds

### Results Files
- `rag_results_*.csv` - RAG predictions per embedding mode
- `tournament_*.csv` - Tournament results per model
- `comparison_metrics.csv` - Cross-method comparison

## Understanding Results

### Performance Categories
1. **Perfect (Rank 1)**: 🟢 Golden answer at top position
2. **Excellent (Rank 2-3)**: 🟢 Very high quality
3. **Good (Rank 4-5)**: 🟡 Acceptable performance
4. **Fair (Rank 6-10)**: 🟠 Found but not prominent
5. **Poor (Rank >10)**: 🔴 Found outside top-10
6. **Not Found**: ⚫ Not in top-K results

### Performance Benchmarks
- **Good Performance**: MRR > 0.3, Hit@1 > 20%, Hit@10 > 60%, Failure rate < 30%
- **Moderate Performance**: MRR: 0.15-0.3, Hit@1: 10-20%, Hit@10: 40-60%, Failure rate: 30-50%
- **Poor Performance**: MRR < 0.15, Hit@1 < 10%, Hit@10 < 40%, Failure rate > 50%

## Architecture

```
source_finding/
├── rag_source_finder.py          # RAG implementation
├── iterative_source_finder.py    # Tournament + Sequential
├── evaluate_source_finding.py    # Evaluation framework
├── rag_visualization.py          # Visualization and analysis
├── source_analysis_v1.ipynb      # Main notebook
├── example_usage.py              # Usage examples
├── results/                      # Output directory
│   ├── rag_results_*.csv
│   ├── tournament_*.csv
│   ├── rag_comprehensive_metrics.csv
│   ├── rag_analysis_summary.txt
│   └── visualizations/
│       ├── 01_performance_heatmap.png
│       ├── 02_hit_at_k_curves.png
│       └── ... (8 total visualizations)
└── README.md                     # This file
```

## Design Decisions

### RAG Approach
- **Embedding Model**: `text-embedding-3-small` (good balance of quality/cost)
- **Similarity**: Cosine similarity (standard for embeddings)
- **Text Construction**: Multiple modes combining name + background + properties
- **Multiple Golden Answers**: Handles cases where multiple sources are valid

### Tournament Approach
- **Batch Size**: 50 sources per batch (default - optimized for speed)
- **Temperature**: 0.2 (low for consistent decisions)
- **Reasoning Capture**: Stores LLM explanation at each round
- **Scalability**: Larger batches = fewer rounds, smaller batches = more reasoning

### Visualization System
- **Modular Design**: Separate classes for data loading, metrics, and visualization
- **Publication Ready**: High-resolution (300 DPI) plots with consistent styling
- **Comprehensive Metrics**: 15+ ranking and performance metrics
- **Backward Compatible**: Handles both old and new CSV formats

## Extending the System

### Adding New Models
Edit `TEST_MODELS` list with models from `easy_llm_importer.py`:

```python
TEST_MODELS = [
    "gpt-4.1-mini",
    "meta-llama-3-1-70b-instruct",
    "deepseek-r1",
    "qwen3-32b",
    # ... add more
]
```

### Adding New Embedding Modes
Extend `RAGSourceFinder.EMBEDDING_MODES` and implement in `_create_embedding_text()`:

```python
def _create_embedding_text(self, name: str, background: str = "", properties: str = "") -> str:
    if self.embedding_mode == "your_new_mode":
        return f"{name}. {your_custom_logic()}"
    # ... existing modes
```

### Adding New Metrics
Extend `RankingMetrics` class:

```python
@staticmethod
def your_custom_metric(df: pd.DataFrame) -> float:
    # Your metric implementation
    return score
```

### Adding New Visualizations
Extend `RAGVisualizer` class:

```python
def plot_your_visualization(self, data, filename="09_your_plot.png"):
    # Your plotting code
    plt.savefig(self.output_dir / filename)
```

## Troubleshooting

### API Key Issues
```python
import os
print("OpenAI:", os.getenv("OPENAI_API_KEY"))
print("DeepInfra:", os.getenv("DEEPINFRA_API_KEY"))
```

### Memory Issues (Large Corpus)
```python
# For RAG, embeddings are ~700 numbers × corpus_size
# Reduce corpus or increase available RAM
```

### Slow Execution
```python
# Use TEST_MODE for quick testing
TEST_MODE = True
TEST_SAMPLE_SIZE = 5  # Very small sample
```

### Visualization Issues
- **No files found**: Check CSV files are named `rag_results_*.csv`
- **Import errors**: Install `pip install seaborn scipy`
- **Missing metrics**: Ensure CSV has `gold_rank` column
- **Blank plots**: Check data loaded correctly

## Performance Characteristics

### RAG Approach
- **Speed**: ~1-2 seconds per query (after corpus embedding)
- **Cost**: ~$0.01-0.05 per 1000 queries (embedding costs)
- **Memory**: ~50-100 MB for 400 examples
- **Accuracy**: MRR ~0.23, Hit@10 ~47% (best modes)

### Tournament Approach
- **Speed**: ~30-60 seconds per query (depends on batch size)
- **Cost**: ~$0.10-0.50 per query (LLM costs)
- **Memory**: ~10-20 MB
- **Accuracy**: Varies by model and batch size

### Visualization System
- **Runtime**: 5-15 seconds for 400 examples × 5 modes
- **Memory**: ~50-100 MB
- **Output**: ~2-3 MB (8 PNGs + CSVs)

## Citation

If using this code, please cite:

```bibtex
@software{source_finding_2024,
  title={Source Finding for Scientific Analogies},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

[Your License Here]

## Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: January 2025  
**Version**: 2.0  
**Status**: Production Ready ✅